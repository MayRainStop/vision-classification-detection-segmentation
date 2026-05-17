import importlib.util
import unittest

from flower_classification.models import build_model


HAS_TORCHVISION = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(HAS_TORCHVISION, "torch and torchvision are required for model factory tests")
class ModelFactoryTests(unittest.TestCase):
    def test_builds_resnet18_cbam_classifier(self):
        model = build_model("resnet18_cbam", num_classes=102, pretrained=False)

        self.assertEqual(model.fc.out_features, 102)

    def test_builds_transformer_classifiers(self):
        vit = build_model("vit_b_16", num_classes=102, pretrained=False)
        swin = build_model("swin_t", num_classes=102, pretrained=False)

        self.assertEqual(vit.heads.head.out_features, 102)
        self.assertEqual(swin.head.out_features, 102)


if __name__ == "__main__":
    unittest.main()
