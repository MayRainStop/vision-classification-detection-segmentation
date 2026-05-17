import unittest

from flower_classification.grid_search import GridValues, build_trial_configs, format_trial_name


class GridSearchTests(unittest.TestCase):
    def test_build_trial_configs_crosses_hyperparameters_for_each_model(self):
        values = GridValues(
            learning_rates=[1e-3, 3e-4],
            weight_decays=[1e-4],
            batch_sizes=[32, 64],
            label_smoothing=[0.0],
        )

        trials = build_trial_configs(
            models=["resnet18", "resnet18_cbam"],
            scratch_models=["resnet18"],
            values=values,
        )

        self.assertEqual(len(trials), 12)
        self.assertEqual(sum(trial.pretrained for trial in trials), 8)
        self.assertEqual(sum(not trial.pretrained for trial in trials), 4)

    def test_format_trial_name_includes_model_pretraining_and_params(self):
        name = format_trial_name(
            model="resnet18",
            pretrained=False,
            learning_rate=3e-4,
            weight_decay=1e-5,
            batch_size=64,
            label_smoothing=0.1,
        )

        self.assertEqual(name, "resnet18_scratch_lr0p0003_wd1e-05_bs64_ls0p1")


if __name__ == "__main__":
    unittest.main()
