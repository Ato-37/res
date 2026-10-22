import base64
import hashlib
import unittest
from io import BytesIO

from PIL import Image

import build_minerva_generation_bias_review as review


def sample_item(item_no=0):
    data = BytesIO()
    Image.new("RGB", (8, 6), (25, 40, 55)).save(data, format="JPEG")
    encoded = base64.b64encode(data.getvalue()).decode("ascii")
    frame_hash = hashlib.sha256(data.getvalue()).hexdigest()
    claim, question = "A person moves a cup.", "What happens to the cup?"
    claim += f" {item_no}"
    frame_hashes = [frame_hash] * review.FRAME_COUNT
    return {
        "item_id": format(item_no, "024x"), "claim": claim, "question": question,
        "frames": [{"rank": i, "source_frame_index": i * 3,
            "time_seconds": i / 2, "data_url": "data:image/jpeg;base64," + encoded}
            for i in range(review.FRAME_COUNT)],
        "source": {"claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "frame_image_sha256": frame_hashes,
            "sample_frame_indices": [i * 3 for i in range(review.FRAME_COUNT)],
            "frame_pts_seconds": [i / 2 for i in range(review.FRAME_COUNT)],
            "frame_image_inventory_sha256": hashlib.sha256(
                __import__("json").dumps(frame_hashes, separators=(",", ":")).encode()
            ).hexdigest()}
    }


class ReviewBuilderTests(unittest.TestCase):
    def test_accepts_exact_64_source_bound_frames(self):
        review.validate_review_items([sample_item(i) for i in range(5)])

    def test_rejects_frame_index_or_image_inventory_tampering(self):
        items = [sample_item(i) for i in range(5)]
        item = items[0]
        item["frames"][17]["source_frame_index"] += 1
        with self.assertRaisesRegex(ValueError, "frame_source_binding_invalid"):
            review.validate_review_items(items)
        items = [sample_item(i) for i in range(5)]
        item = items[0]
        item["frames"].pop()
        with self.assertRaisesRegex(ValueError, "item_invalid"):
            review.validate_review_items(items)

    def test_page_fingerprint_is_embedded_and_label_blind(self):
        item = [sample_item(i) for i in range(5)]
        provenance = {"builder_sha256": "b" * 64,
            "source_manifest_sha256": "c" * 64}
        raw = review._render(item, provenance).encode("utf-8")
        finalized, fingerprint = review._finalize_page(raw)
        self.assertIn(fingerprint.encode(), finalized)
        self.assertNotIn(b"bbox_key_bias", finalized)
        self.assertNotIn(b"spatial_sham_key_bias", finalized)
        self.assertIn(b"respondents.push", finalized)
        self.assertIn(b"artifact_fingerprint", finalized)
        tampered = finalized.replace(b"<title>", b"<title>Tampered")
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(tampered).hexdigest())

    def test_requires_five_items_for_final_review(self):
        with self.assertRaisesRegex(ValueError, "requires_five_items"):
            review.validate_review_items([sample_item()] * 4)


if __name__ == "__main__":
    unittest.main()
