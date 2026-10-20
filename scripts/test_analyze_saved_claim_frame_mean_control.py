from __future__ import annotations
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from analyze_saved_claim_frame_mean_control import analyze, audit_provenance

def make(token,value):
    return {"status":"completed_one_row_posthoc_frame_mean_sensitivity","post_result_exploration":True,
        "direct_faithfulness":False,"human_review":{"review_actually_performed":False},
        "requested_forwards":2,"attempted_forwards":2,"completed_forwards":2,
        "row_token_sha256":token*64,"branch_target_scores":{"original_video":{"target_logprob_mean":value-1},
            "per_frame_rgb_mean_video":{"target_logprob_mean":-1}},
        "primary_descriptive_contrast_original_minus_frame_mean_nats_per_token":value}

class FrameMeanSummaryTests(unittest.TestCase):
    def test_statistics_and_direct_faithfulness_flag(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for i,v in enumerate((0.3,-0.1,0.8)):
                p=Path(tmp)/f"{i}.json";p.write_text(json.dumps(make(str(i+1),v)));paths.append(p)
            result=analyze(tuple(paths))
        self.assertAlmostEqual(result["finite_cohort_mean_nats_per_token"],1/3)
        self.assertEqual(result["finite_cohort_range_nats_per_token"],[-0.1,0.8])
        self.assertFalse(result["direct_faithfulness"])

    def test_incomplete_or_duplicate_receipts_rejected(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for i in range(3):
                p=Path(tmp)/f"{i}.json";p.write_text(json.dumps(make(str(i+1),0.2)));paths.append(p)
            paths[2].write_text(json.dumps(make("1",0.2)))
            with self.assertRaisesRegex(ValueError,"duplicated"): analyze(tuple(paths))
            paths[2].write_text(json.dumps(make("3",0.2)))
            x=make("3",0.2);x["completed_forwards"]=1;paths[2].write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,"complete"): analyze(tuple(paths))

    def test_provenance_audit_binds_receipt_packet_row_and_frozen_filter(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);private=root/"private";private.mkdir()
            rows=[];assignments=[];pairs=[]
            for i in range(3):
                video=private/f"video{i}.bin";video.write_bytes(f"local-video-{i}".encode())
                media=hashlib.sha256(video.read_bytes()).hexdigest();qid=f"sample-{i}"
                qid_hash=hashlib.sha256(qid.encode()).hexdigest();claim=f"saved claim {i}"
                pair=chr(ord("a")+i)*64;pairs.append(pair)
                rows.append({"video_sha256":media,"dataset_question_id":qid,
                    "candidate_pair_sha256":pair,"generated_parsed":{"claim":claim},
                    "local_video_path":str(video)})
                assignments.append({"media_sha256":media,"qid_sha256":qid_hash})
            selected={"schema_version":1,"pair_sha256":pairs}
            selected_path=private/"selected.json";selected_path.write_text(json.dumps(selected))
            batch={"schema_version":1,"pair_sha256":pairs}
            batch_path=private/"batch.json";batch_path.write_text(json.dumps(batch))
            packet={"status":"passed","dataset_split":"test",
                "candidate_pair_filter_sha256":hashlib.sha256(batch_path.read_bytes()).hexdigest(),
                "records":rows}
            packet_path=private/"packet.json";packet_path.write_text(json.dumps(packet))
            lock={"dataset_split":"test","status":"frozen_before_claim_generation_and_scoring",
                "selected_pair_filter_sha256":hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                "assignments":assignments}
            lock_path=private/"lock.json";lock_path.write_text(json.dumps(lock))
            script=Path(__file__).resolve();repo=script.parents[1]
            receipt_paths=[]
            for i,row in enumerate(rows):
                qid=row["dataset_question_id"];media=row["video_sha256"]
                token=hashlib.sha256((media+qid).encode()).hexdigest()
                receipt={"private_input_packet_name":packet_path.name,
                    "private_input_packet_sha256":hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                    "packet_record_index_zero_based":i,"row_token_sha256":token,
                    "saved_claim_text_sha256":hashlib.sha256(row["generated_parsed"]["claim"].encode()).hexdigest(),
                    "source_video_sha256":media,"human_review":{"review_actually_performed":False},
                    "source_sha256":{str(script.relative_to(repo)):
                        hashlib.sha256(script.read_bytes()).hexdigest()}}
                receipt_path=private/f"receipt{i}.json";receipt_path.write_text(json.dumps(receipt))
                receipt_paths.append(receipt_path)
            result=audit_provenance(receipt_paths,lock_path,selected_path,
                {packet_path.name:packet_path},{packet_path.name:batch_path})
            self.assertEqual(result["status"],"provenance_verified")
            self.assertEqual(result["source_hash_checks"],3)
            lock["selected_pair_filter_sha256"]="0"*64;lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError,"selected_filter_hash_mismatch"):
                audit_provenance(receipt_paths,lock_path,selected_path,
                    {packet_path.name:packet_path},{packet_path.name:batch_path})

if __name__=="__main__":unittest.main()
