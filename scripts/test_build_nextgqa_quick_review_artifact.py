import json
import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_nextgqa_quick_review_artifact import (
    DEFAULT_OUTPUT, browser_compatible_bytes, build, exploratory_item_id, load_translations,
    private_output_path, serialize_inline_json, translation_key,
)


class QuickReviewArtifactSafetyTests(unittest.TestCase):
    def test_default_output_targets_current_simple_v14_artifact(self):
        self.assertEqual(
            DEFAULT_OUTPUT.name,
            "quick_review_ja_simple_20261019_itemid_v14.html",
        )
        self.assertTrue(DEFAULT_OUTPUT.is_relative_to(
            Path(__file__).resolve().parents[1] / "data/private_nextgqa"
        ))

    def test_item_id_is_stable_opaque_and_does_not_echo_review_token(self):
        token = "private-review-token-123"
        first = exploratory_item_id(token)
        self.assertEqual(first, exploratory_item_id(token))
        self.assertNotEqual(first, exploratory_item_id(token + "x"))
        self.assertEqual(len(first), 20)
        self.assertNotIn(token, first)

    def test_inline_json_cannot_close_script_element(self):
        payload = {"question": "</script><script>alert('x')</script>"}
        serialized = serialize_inline_json(payload)
        self.assertNotIn("<", serialized)
        self.assertEqual(json.loads(serialized), payload)

    def test_output_must_stay_inside_private_data_directory(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ValueError):
                private_output_path(Path(outside) / "review.html")
        accepted = private_output_path(
            Path(__file__).resolve().parents[1]
            / "data/private_nextgqa/quick_review_test.html"
        )
        self.assertTrue(accepted.is_relative_to(Path(__file__).resolve().parents[1] / "data/private_nextgqa"))

    def test_browser_video_conversion_removes_audio_and_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")

            probe_calls = 0

            def fake_run(args, **kwargs):
                nonlocal probe_calls
                if args[0] == "ffprobe":
                    probe_calls += 1
                if args[0] == "ffprobe" and probe_calls == 1:
                    return type("Result", (), {"stdout": json.dumps({"streams": [
                        {"codec_type": "video", "codec_name": "h264"},
                        {"codec_type": "audio", "codec_name": "aac"},
                        {"codec_type": "subtitle", "codec_name": "mov_text"},
                    ]})})()
                if args[0] == "ffmpeg":
                    self.assertIn("-an", args)
                    self.assertIn("-sn", args)
                    self.assertEqual(args[args.index("-map") + 1], "0:v:0")
                    Path(args[-1]).write_bytes(b"video-only")
                    return type("Result", (), {"stdout": ""})()
                return type("Result", (), {"stdout": json.dumps({"streams": [
                    {"codec_type": "video", "codec_name": "h264"},
                ]})})()

            with patch("build_nextgqa_quick_review_artifact.subprocess.run", side_effect=fake_run):
                self.assertEqual(browser_compatible_bytes(source), b"video-only")

    def test_translation_key_does_not_reveal_source_text(self):
        question = "private source question"
        claim = "private source claim"
        key = translation_key(question, claim)
        self.assertEqual(len(key), 64)
        self.assertNotIn(question, key)
        self.assertNotIn(claim, key)
        self.assertNotEqual(key, translation_key(question, "changed claim"))

    def test_translation_json_requires_exact_schema_and_pair_hash_key(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "data/private_nextgqa") as tmp:
            path = Path(tmp) / "translations.json"
            key = translation_key("source question", "source claim")
            path.write_text(json.dumps({"schema_version": 1, "items": {
                key: {"question_ja": "質問？", "claim_ja": "説明。"}
            }}), encoding="utf-8")
            self.assertEqual(load_translations(path)[key], ("質問？", "説明。"))
            path.write_text(json.dumps({"schema_version": 1.0, "items": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_translations(path)

    def test_unknown_question_is_rejected_instead_of_silently_shown_in_english(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "data/private_nextgqa") as tmp:
            package = Path(tmp) / "package"
            package.mkdir()
            output = Path(tmp) / "review.html"
            records = [{"review_token": f"tok{i}", "question": f"known {i}", "claim_text": "claim"}
                       for i in range(2)]
            records.append({"review_token": "tok2", "question": "new item", "claim_text": "claim"})
            (package / "index.html").write_text(
                '<script id="data" type="application/json">'
                + json.dumps({"records": records}) + "</script>", encoding="utf-8"
            )
            for i in range(3):
                (package / f"tok{i}.mp4").write_bytes(b"mock")
            with patch("build_nextgqa_quick_review_artifact.browser_compatible_bytes", return_value=b"mp4"):
                with self.assertRaisesRegex(ValueError, "日本語訳がありません"):
                    build(package, output, translations={
                        translation_key("known 0", "claim"): ("質問0", "説明0"),
                        translation_key("known 1", "claim"): ("質問1", "説明1"),
                    })
            self.assertFalse(output.exists())

    def test_rendered_page_goes_from_nickname_to_simple_questions_without_file_picker(self):
        question, question_ja, claim, claim_ja = (
            "private source question", "日本語の質問？", "private source claim", "日本語の説明。"
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "data/private_nextgqa") as tmp:
            package = Path(tmp) / "package"
            package.mkdir()
            output = Path(tmp) / "review.html"
            records = [{"review_token": f"tok{i}", "question": f"private question {i}", "claim_text": f"private claim {i}"}
                       for i in range(3)]
            records[0] = {"review_token": "tok0", "question": "__ITEMS__ " + question, "claim_text": claim}
            (package / "index.html").write_text(
                '<script id="data" type="application/json">'
                + json.dumps({"records": records}) + "</script>", encoding="utf-8"
            )
            for i in range(3):
                (package / f"tok{i}.mp4").write_bytes(b"mock")
            with patch("build_nextgqa_quick_review_artifact.browser_compatible_bytes", return_value=b"mp4"):
                build(package, output, translations={
                    translation_key("__ITEMS__ " + question, claim): (question_ja, claim_ja),
                    translation_key("private question 1", "private claim 1"): ("質問1", "説明1"),
                    translation_key("private question 2", "private claim 2"): ("質問2", "説明2"),
                })
            rendered = output.read_text(encoding="utf-8")
            self.assertIn(question_ja, rendered)
            self.assertIn(claim_ja, rendered)
            self.assertIn("id=\"nickname\"", rendered)
            self.assertIn("ニックネームを入力", rendered)
            self.assertIn("質問へ", rendered)
            self.assertIn("回答は外へ送信されません", rendered)
            self.assertIn("必要なら最後に、この端末へJSONで保存できます", rendered)
            self.assertIn("ファイル選択や時刻入力はありません", rendered)
            self.assertIn("厳密な幻覚判定や研究上の性能評価ではありません", rendered)
            self.assertIn("質問を読んで、何を答えればよいか分かりますか？", rendered)
            self.assertIn("動画を見て、説明に書かれた答えは正しいと思いますか？", rendered)
            self.assertIn("説明は、答えの根拠となる動画中の出来事（必要なら動きの順序）を十分に示していますか？", rendered)
            self.assertIn("②は動画と見比べた答えの正しさ、③は説明が動画中の出来事を根拠として示しているか", rendered)
            self.assertIn("質問の意味や動画の場面がはっきりしないときは推測せず", rendered)
            self.assertIn("①は「わかりにくい」、②③は「判断できない」を選べます", rendered)
            self.assertNotIn("答えが質問に合っていても", rendered)
            self.assertIn('name="question_clarity"', rendered)
            self.assertIn('name="answer_question_fit"', rendered)
            self.assertIn('name="video_evidence_sufficiency"', rendered)
            self.assertIn("足りない", rendered)
            self.assertNotIn('type="file"', rendered)
            self.assertNotIn('id="time"', rendered)
            self.assertNotIn('id="note"', rendered)
            self.assertIn("item.question_ja", rendered)
            self.assertIn("item.claim_ja", rendered)
            self.assertIn("元の英語の質問・説明を見る", rendered)
            self.assertIn('id="prompt-source"', rendered)
            self.assertIn('id="claim-source"', rendered)
            self.assertIn("$('prompt-source').textContent=item.question", rendered)
            self.assertIn("$('claim-source').textContent=item.claim", rendered)
            show_question_start = rendered.index("function showQuestion()")
            show_question_end = rendered.index("$('welcome-form').addEventListener", show_question_start)
            show_question_script = rendered[show_question_start:show_question_end]
            self.assertIn("$('source-language').open=false", show_question_script)
            self.assertIn("__ITEMS__ private source question", rendered)
            self.assertIn('id="summary-count"', rendered)
            self.assertIn('id="summary-items"', rendered)
            self.assertIn("function showSummary()", rendered)
            self.assertIn("${current.answers.length} / ${items.length} 件に回答済み", rendered)
            self.assertIn("$('summary-items').replaceChildren", rendered)
            self.assertIn("$('summary').classList.remove('hidden')", rendered)
            summary_start = rendered.index("function showSummary()")
            summary_end = rendered.index("function prepareNextReviewer()", summary_start)
            summary_script = rendered[summary_start:summary_end]
            self.assertIn("li.textContent=", summary_script)
            self.assertNotIn("innerHTML", summary_script)
            self.assertNotIn("nickname", summary_script)
            self.assertNotIn("answer.note", summary_script)
            self.assertNotIn("item_id", summary_script)
            self.assertIn("正式なレビューやfaithfulness評価ではありません", rendered)
            self.assertIn("全員分を保存", rendered)
            self.assertIn("次の人へ", rendered)
            self.assertIn('id="saved-file"', rendered)
            self.assertIn("JSONをもう一度ダウンロード", rendered)
            self.assertIn("const respondents=[]", rendered)
            self.assertIn("respondents.push({nickname,answers:answers.slice()})", rendered)
            self.assertIn("function prepareNextReviewer()", rendered)
            self.assertIn("${nextNumber}人目のニックネームを入力", rendered)
            self.assertIn("$('cancel-next').classList.remove('hidden')", rendered)
            self.assertIn("$('next-reviewer').addEventListener('click',prepareNextReviewer)", rendered)
            self.assertIn("$('cancel-next').addEventListener('click'", rendered)
            self.assertIn("$('next-reviewer').addEventListener('click'", rendered)
            self.assertIn("schema_version:6", rendered)
            self.assertIn("artifact_fingerprint:document.querySelector", rendered)
            self.assertRegex(rendered, r'name="review-artifact-sha256" content="[0-9a-f]{64}"')
            self.assertNotIn("__ARTIFACT_FINGERPRINT__", rendered)
            fingerprint = re.search(
                r'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">', rendered
            ).group(1)
            canonical = rendered.replace(fingerprint, "__ARTIFACT_FINGERPRINT__", 1)
            self.assertEqual(fingerprint, hashlib.sha256(canonical.encode()).hexdigest())
            self.assertIn("JSONには各人のニックネームが含まれます", rendered)
            self.assertIn("自分のニックネームを入力してから質問に進んでください", rendered)
            self.assertIn("ファイルを共有・公開しないでください", rendered)
            self.assertIn("ダウンロードを開始しました", rendered)
            self.assertIn("ブラウザのダウンロード一覧で保存を確認してください", rendered)
            self.assertIn("if(exportUrl)URL.revokeObjectURL(exportUrl)", rendered)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
