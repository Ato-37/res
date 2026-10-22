from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from argparse import Namespace
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from audit_nextgqa_review_release_order import _ROSTER_DOMAIN
from nextgqa_capture_operator_attestation import (
    CaptureOperatorAttestationError, create_attestation, verify_attestation,
)
from nextgqa_pre_review_integrity_receipt import _canonical, _public_der, _sha, _sign
from run_nextgqa_capture_operator_attestation import main as main_attestation_cli, run as run_private_cli


class CaptureAttestationTests(unittest.TestCase):
    """All fixture artifacts are synthetic bytes under a private temp directory."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.keys = {}
        for name in ("root", "owner", "claim", "frame", "capture", "custodian", "prediction", "witness"):
            priv, pub = self.root / f"{name}.priv", self.root / f"{name}.pub"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(priv)], check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(priv), "-pubout", "-out", str(pub)], check=True, capture_output=True)
            priv.chmod(0o600); pub.chmod(0o600)
            self.keys[name] = (priv.read_bytes(), pub.read_bytes())
        self.root_pin = _sha(_public_der(self.keys["root"][1]))
        self.integrity = self._signed_roster("nextgqa_pre_review_trust_roster_v1", "nextgqa_pre_review_roster_v1\0".encode(), [
            self._entry("freeze_owner", "owner", "owner-key"),
            self._entry("claim_reviewer", "claim", "claim-key"),
            self._entry("frame_reviewer", "frame", "frame-key")])
        self.release = self._release_roster()
        self.inputs = self._inputs()
        self.attestation = create_attestation(reviewer_private_key_pem=self.keys["capture"][0], reviewer_public_key_pem=self.keys["capture"][1], **self.inputs)

    def tearDown(self): self.temp.cleanup()

    def _entry(self, role, keyname, key_id):
        pem = self.keys[keyname][1]
        return {"role": role, "key_id": key_id,
                "public_key_pem_base64": base64.b64encode(pem).decode(),
                "public_key_der_sha256": _sha(_public_der(pem))}

    def _signed_roster(self, purpose, domain, entries):
        body = {"schema_version": 1, "purpose": purpose, "roster_id": "synthetic-roster", "keys": entries}
        sig = _sign(self.keys["root"][0], self.keys["root"][1], domain + _canonical(body))
        return _canonical({**body, "signature_ed25519_base64": base64.b64encode(sig).decode()}) + b"\n"

    def _release_roster(self, capture="capture"):
        return self._signed_roster("nextgqa_review_release_roster_v1", _ROSTER_DOMAIN, [
            self._entry("freeze_owner", "owner", "owner-key"), self._entry("custodian", "owner", "owner-key"),
            self._entry("capture_operator", capture, "capture-key"), self._entry("claim_reviewer", "claim", "claim-key"),
            self._entry("frame_reviewer", "frame", "frame-key"), self._entry("prediction_link_reviewer", "prediction", "prediction-key"),
            self._entry("release_witness", "witness", "witness-key")])

    def _inputs(self):
        tokens = ["A" * 24, "B" * 24, "C" * 24]
        anchors = ("fit_begin", "heldout_quarter", "fit_midpoint", "heldout_three_quarter", "fit_end")
        records = []
        for i, token in enumerate(tokens):
            records.append({"review_token": token, "source_media_sha256": f"{i+1:064x}",
                "fit_anchors": [{"anchor_id": a, "decoder_pts_seconds": str(j+1)} for j,a in enumerate((anchors[0], anchors[2], anchors[4]))],
                "heldout_anchors": [{"anchor_id": a, "decoder_pts_seconds": str(j+2)} for j,a in enumerate((anchors[1], anchors[3]))]})
        plan = _canonical({"schema_version":1,"purpose":"predeclared_html5_clock_pts_anchor_plan","status":"predeclared_frozen","execution_authorization":"not_granted","records":records})+b"\n"
        plan_sha = _sha(plan)
        prov = _canonical({"schema_version":1,"purpose":"private_clock_anchor_plan_candidate_provenance","status":"candidate_not_jointly_frozen","execution_authorization":"not_granted","plan_sha256":plan_sha})+b"\n"
        prov_sha = _sha(prov)
        observations = {"schema_version":3,"purpose":"local_unvalidated_html5_player_clock_observations","status":"raw_player_clock_observations_only","execution_authorization":"not_granted","records":[]}
        for token in tokens:
            observations["records"].append({"review_token":token,"events":[{"anchor_id":a,"sequence":n,"clock":"synthetic"} for n,a in enumerate(anchors,1)]})
        obs_raw = _canonical(observations)+b"\n"
        idx = _canonical({"schema_version":1,"purpose":"unvalidated_html5_clock_observation_integrity_index","payload_sha256":_sha(obs_raw),"created_at_utc":"2026-10-18T00:00:00Z","status":"raw_observations_only","execution_authorization":"not_granted"})+b"\n"
        screenshots = {}; crops = {}; receipt_rows = []; cross_rows = []
        for token in tokens:
            for n,a in enumerate(anchors,1):
                name=f"{token}_{a}.png"; shot=f"synthetic screenshot {token} {a}".encode()
                screenshots[name]=shot
                receipt_rows.append({"review_token":token,"anchor_id":a,"screenshot_file":name,"screenshot_sha256":_sha(shot),"screenshot_captured_at_utc":"2026-10-18T00:00:00Z","capture_method":"windows_gdi_copyfromscreen_foreground_window_v1","foreground_process":"synthetic","foreground_pid":1,"foreground_window_title":"synthetic","foreground_window_rect_xywh":[0,0,10,10],"virtual_screen_rect_xywh":[0,0,10,10],"hotkey_message_tick_ms":n,"capture_complete_tick_ms":n+1,"hotkey_to_capture_ms":1})
        receipt = _canonical({"schema_version":1,"purpose":"private_windows_compositor_screenshot_capture_receipt_candidate","status":"self_reported_capture_receipt_not_independently_authenticated","execution_authorization":"not_granted","plan_sha256":plan_sha,"plan_provenance_sha256":prov_sha,"capture_method":"windows_gdi_copyfromscreen_foreground_window_v1","capture_helper_file_sha256":"f"*64,"created_at_utc":"2026-10-18T00:00:00Z","records":receipt_rows})+b"\n"
        cropmap = b"synthetic crop map bytes\n"
        for row in receipt_rows:
            cropname="crop_"+row["screenshot_file"]
            crop=f"synthetic crop {row['review_token']} {row['anchor_id']}".encode(); crops[cropname]=crop
            cross_rows.append({"review_token":row["review_token"],"anchor_id":row["anchor_id"],"screenshot_file":row["screenshot_file"],"screenshot_sha256":row["screenshot_sha256"],"bbox_xywh":[0,0,1,1],"crop_file":cropname,"crop_sha256":_sha(crop),"crop_rgb_dimensions":[1,1],"crop_purity":"human_review_required"})
        cross = _canonical({"schema_version":1,"purpose":"private_candidate_screenshot_crop_crosswalk","status":"candidate_only","execution_authorization":"not_granted","plan_sha256":plan_sha,"plan_provenance_sha256":prov_sha,"capture_receipt_sha256":_sha(receipt),"crop_map_sha256":_sha(cropmap),"bbox_video_only_purity":"not_established_human_review_required","records":cross_rows})+b"\n"
        # Materialize and read back synthetic artifacts only, with restrictive modes.
        for folder, mapping in (("screenshots",screenshots),("crops",crops)):
            d=self.root/folder; d.mkdir(mode=0o700)
            for filename,raw in mapping.items():
                p=d/filename; p.write_bytes(raw); p.chmod(0o600); mapping[filename]=p.read_bytes()
        return {"plan_raw":plan,"expected_plan_sha256":plan_sha,"plan_provenance_raw":prov,"expected_plan_provenance_sha256":prov_sha,"observations_raw":obs_raw,"observation_index_raw":idx,"capture_receipt_raw":receipt,"crop_map_raw":cropmap,"crop_crosswalk_raw":cross,"screenshot_files":screenshots,"crop_files":crops,"integrity_roster_raw":self.integrity,"release_roster_raw":self.release,"root_public_key_pem":self.keys["root"][1],"expected_root_public_der_sha256":self.root_pin}

    def test_roundtrip_is_exact_byte_binding_and_non_authorizing(self):
        report=verify_attestation(attestation_raw=self.attestation, **self.inputs)
        self.assertEqual(report["anchor_count"],15)
        for key in ("human_identity_verified","capture_process_authenticated","frame_correspondence_reviewed","score_ready","model_load_permitted"):
            self.assertFalse(report[key])
        self.assertEqual(report["execution_authorization"],"not_granted")

    def test_tampered_screenshot_or_crop_fails(self):
        for field, name in (("screenshot_files",next(iter(self.inputs["screenshot_files"]))),("crop_files",next(iter(self.inputs["crop_files"])) )):
            changed=dict(self.inputs); changed[field]=dict(changed[field]); changed[field][name]+=b"tamper"
            with self.subTest(field=field), self.assertRaises(CaptureOperatorAttestationError):
                verify_attestation(attestation_raw=self.attestation,**changed)

    def test_plan_hash_and_raw_event_index_are_bound(self):
        changed=dict(self.inputs); changed["expected_plan_sha256"]="0"*64
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"frozen_anchor_plan_hash_mismatch"):
            verify_attestation(attestation_raw=self.attestation,**changed)
        changed=dict(self.inputs); changed["observations_raw"]+=b" "
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"observation_index_binding_invalid"):
            verify_attestation(attestation_raw=self.attestation,**changed)

    def test_event_and_os_receipt_bytes_are_bound(self):
        obj=json.loads(self.inputs["observations_raw"]); obj["records"][0]["events"][0]["clock"]="tampered"
        changed=dict(self.inputs); changed["observations_raw"]=_canonical(obj)+b"\n"
        idx=json.loads(changed["observation_index_raw"]); idx["payload_sha256"]=_sha(changed["observations_raw"])
        changed["observation_index_raw"]=_canonical(idx)+b"\n"
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"attestation_binding_mismatch"):
            verify_attestation(attestation_raw=self.attestation,**changed)
        changed=dict(self.inputs); changed["capture_receipt_raw"]+=b" "
        with self.assertRaises(CaptureOperatorAttestationError):
            verify_attestation(attestation_raw=self.attestation,**changed)

    def test_bool_anchor_count_rejected(self):
        obj=json.loads(self.attestation); obj["anchor_count"]=True
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"attestation_schema_invalid"):
            verify_attestation(attestation_raw=_canonical(obj),**self.inputs)

    def test_role_key_reuse_rejected(self):
        bad=self._release_roster(capture="claim")
        changed=dict(self.inputs); changed["release_roster_raw"]=bad
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"release_roster_invalid"):
            create_attestation(reviewer_private_key_pem=self.keys["capture"][0],reviewer_public_key_pem=self.keys["capture"][1],**changed)

    def _cli_args(self, *, output="attestation.json", **overrides):
        filenames={"plan_raw":"plan.json","plan_provenance_raw":"provenance.json",
            "observations_raw":"observations.json","observation_index_raw":"observation-index.json",
            "capture_receipt_raw":"capture-receipt.json","crop_map_raw":"crop-map.json",
            "crop_crosswalk_raw":"crop-crosswalk.json","integrity_roster_raw":"integrity.json",
            "release_roster_raw":"release.json"}
        for key,name in filenames.items():
            p=self.root/name; p.write_bytes(self.inputs[key]); p.chmod(0o600)
        p=self.root/"root-public.pem"; p.write_bytes(self.keys["root"][1]); p.chmod(0o600)
        p=self.root/"operator-private.pem"; p.write_bytes(self.keys["capture"][0]); p.chmod(0o600)
        p=self.root/"operator-public.pem"; p.write_bytes(self.keys["capture"][1]); p.chmod(0o600)
        values={"private_root":str(self.root),"plan":"plan.json","plan_sha256":self.inputs["expected_plan_sha256"],
            "plan_provenance":"provenance.json","plan_provenance_sha256":self.inputs["expected_plan_provenance_sha256"],
            "observations":"observations.json","observation_index":"observation-index.json",
            "capture_receipt":"capture-receipt.json","crop_map":"crop-map.json","crop_crosswalk":"crop-crosswalk.json",
            "screenshots_dir":"screenshots","crops_dir":"crops","integrity_roster":"integrity.json",
            "release_roster":"release.json","root_public_key":"root-public.pem","root_public_der_sha256":self.root_pin,
            "operator_private_key":"operator-private.pem","operator_public_key":"operator-public.pem","output":output}
        values.update(overrides)
        return Namespace(**values)

    def test_private_cli_creates_mode_600_exclusive_output_and_content_free_status(self):
        from contextlib import redirect_stdout
        from io import StringIO
        out=StringIO()
        with redirect_stdout(out): run_private_cli(self._cli_args())
        self.assertEqual(out.getvalue(),"status=candidate_capture_operator_attestation_created authorization=not_granted model_load_permitted=false\n")
        target=self.root/"attestation.json"
        self.assertEqual(target.stat().st_mode & 0o777,0o600)
        self.assertEqual(json.loads(target.read_bytes())["anchor_count"],15)
        with self.assertRaises(FileExistsError): run_private_cli(self._cli_args())

    def test_private_cli_rejects_outside_path_and_symlinked_input(self):
        args=self._cli_args(plan="../plan.json")
        with self.assertRaisesRegex(ValueError,"path_must_be_relative_and_contained"):
            run_private_cli(args)
        args=self._cli_args()
        (self.root/"plan.json").unlink(); (self.root/"plan.json").symlink_to("provenance.json")
        with self.assertRaises(OSError): run_private_cli(args)

    def test_private_cli_rejects_extra_directory_artifact(self):
        args=self._cli_args()
        extra=self.root/"screenshots"/"unexpected.png"; extra.write_bytes(b"extra"); extra.chmod(0o600)
        with self.assertRaisesRegex(ValueError,"artifact_directory_contents_invalid"):
            run_private_cli(args)
        self.assertFalse((self.root/"attestation.json").exists())

    def test_private_cli_rejects_intermediate_and_leaf_symlink_escapes(self):
        args=self._cli_args()
        external=tempfile.TemporaryDirectory(dir=self.root.parent); self.addCleanup(external.cleanup)
        outside=Path(external.name)
        inner=outside/"inner"; inner.mkdir(parents=True,mode=0o700); outside.chmod(0o700); inner.chmod(0o700)
        for name,raw in self.inputs["screenshot_files"].items():
            p=inner/name; p.write_bytes(raw); p.chmod(0o600)
        (self.root/"redirect").symlink_to(outside, target_is_directory=True)
        args.screenshots_dir="redirect/inner"
        with self.assertRaises(OSError): run_private_cli(args)
        (self.root/"redirect").unlink()
        screen_dir=self.root/"screenshots"
        first=next(iter(self.inputs["screenshot_files"]))
        target=screen_dir/first; target.unlink(); target.symlink_to(inner/first)
        args=self._cli_args()
        # _cli_args rewrites the target through the symlink; replace it afterward.
        target=screen_dir/first; target.unlink(); target.symlink_to(inner/first)
        with self.assertRaises(OSError): run_private_cli(args)

    def test_malformed_receipt_and_crosswalk_fail_with_sanitized_status(self):
        for filename in ("capture-receipt.json","crop-crosswalk.json"):
            args=self._cli_args(output="never-created.json")
            p=self.root/filename; p.write_bytes(b"[]\n"); p.chmod(0o600)
            argv=["run_nextgqa_capture_operator_attestation"]
            for key,value in vars(args).items():
                argv.extend(["--"+key.replace("_","-"),str(value)])
            stderr=StringIO()
            with patch.object(sys,"argv",argv), redirect_stderr(stderr):
                self.assertEqual(main_attestation_cli(),2)
            self.assertEqual(stderr.getvalue(),"status=failed code=private_attestation_error\n")
            self.assertFalse((self.root/"never-created.json").exists())

    def test_missing_or_duplicate_anchor_artifact_fails_closed(self):
        changed=dict(self.inputs); changed["screenshot_files"]=dict(changed["screenshot_files"]); changed["screenshot_files"].pop(next(iter(changed["screenshot_files"])))
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"screenshot_artifact_set_mismatch"):
            verify_attestation(attestation_raw=self.attestation,**changed)
        changed=dict(self.inputs); changed["crop_files"]=dict(changed["crop_files"]); changed["crop_files"]["unexpected.png"]=b"extra"
        with self.assertRaisesRegex(CaptureOperatorAttestationError,"crop_artifact_set_mismatch"):
            verify_attestation(attestation_raw=self.attestation,**changed)
        duplicate=json.loads(self.inputs["crop_crosswalk_raw"]); duplicate["records"][1]=duplicate["records"][0]
        changed=dict(self.inputs); changed["crop_crosswalk_raw"]=_canonical(duplicate)
        with self.assertRaises(CaptureOperatorAttestationError): verify_attestation(attestation_raw=self.attestation,**changed)


if __name__ == "__main__": unittest.main()
