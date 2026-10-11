"""CPU-only diagnostic of installed Qwen temporal coordinate quantization.

Calls the actual installed model methods without instantiating model weights.
Compares only temporal coordinates to the fractional rule in official upstream
source. Text continuation is reported separately, not declared a defect.
Usage: PYTHONPATH=scripts .venv/bin/python scripts/audit_qwen_temporal_positions.py --output results/NEW.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo / 'results') or output.exists():
        parser.error('New output under results/ required; overwrite refused')
    import torch
    import transformers
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    fixture = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=2, tokens_per_second=2)))
    fixture.get_vision_position_ids = MethodType(Qwen2_5_VLModel.get_vision_position_ids, fixture)
    cases = []
    # Fixed before diagnostic outcomes: subsecond, integer, fractional, and the
    # step from the previously published real-video artifact. No fitted values.
    for steps in (2, 4):
        ids = torch.tensor([[11] + [151656]*steps + [12]])
        modalities = torch.tensor([[0] + [2]*steps + [0]])
        grid = torch.tensor([[steps,2,2]])
        for seconds in (0.5,0.75,0.9999,1.0,1.5,2.0,199.9998016357422):
            pos, delta = Qwen2_5_VLModel.get_rope_index(fixture, input_ids=ids,
                mm_token_type_ids=modalities, video_grid_thw=grid,
                second_per_grid_ts=[seconds])
            observed = pos[0,0,1:1+steps].tolist()
            relative = [x-observed[0] for x in observed]
            fractional = (torch.arange(steps)*seconds*2).long().tolist()
            cases.append({'grid_steps':steps,'seconds_per_grid':seconds,
                'observed_relative_temporal_coordinates':relative,
                'fractional_rule_relative_coordinates':fractional,
                'temporal_coordinates_equal':relative==fractional,
                'observed_distinct_temporal_coordinates':len(set(relative)),
                'post_video_text_coordinate':pos[:,0,-1].tolist(),
                'max_video_coordinate':int(pos[:,0,1:1+steps].max()),
                'rope_delta':int(delta[0,0])})
    result = {
        'schema_version':1,'purpose':'installed_qwen_position_id_cpu_diagnostic',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'runner_sha256':digest(__file__),
        'installed_model_source_sha256':digest(inspect.getfile(Qwen2_5_VLModel)),
        'installed_processor_source_sha256':digest(inspect.getfile(Qwen2_5_VLProcessor)),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__},
        'fixture_config':{'spatial_merge_size':2,'tokens_per_second':2,'batch_size':1,
            'text_tokens_before_video':1,'text_tokens_after_video':1,'spatial_grid':[2,2]},
        'reference_source':'https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py',
        'original_fractional_rule_source':'https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py',
        'requested_cases':14,'effective_cases':len(cases),'failures':0,
        'temporal_coordinate_mismatch_cases':sum(not c['temporal_coordinates_equal'] for c in cases),
        'cases':cases,
        'limitations':['Synthetic token fixtures; no media/model weights/CUDA inference.',
            'Single video block, minimal spatial grid, positive timestamps only.',
            'Distinct positions and numerical consistency do not prove semantic temporal evidence.',
            'Text continuation is version-specific source behavior, not labeled a validated defect.',
            'No efficacy effect, uncertainty interval or statistical hypothesis test.']
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'CPU temporal-coordinate diagnostic: {len(cases)}/14 cases, '
        f'{result["temporal_coordinate_mismatch_cases"]} differ from fractional rule; no model inference.')


if __name__ == '__main__':
    main()
