import unittest

from flower_classification.train import EarlyStoppingState


class EarlyStoppingStateTests(unittest.TestCase):
    def test_stops_after_patience_epochs_without_improvement(self):
        state = EarlyStoppingState(patience=2)

        first = state.step(score=0.50, epoch=1)
        second = state.step(score=0.49, epoch=2)
        third = state.step(score=0.48, epoch=3)

        self.assertTrue(first.improved)
        self.assertFalse(second.should_stop)
        self.assertTrue(third.should_stop)
        self.assertEqual(state.best_epoch, 1)

    def test_resets_wait_count_when_score_improves(self):
        state = EarlyStoppingState(patience=2)

        state.step(score=0.50, epoch=1)
        state.step(score=0.49, epoch=2)
        improved = state.step(score=0.51, epoch=3)
        after_improvement = state.step(score=0.50, epoch=4)

        self.assertTrue(improved.improved)
        self.assertEqual(state.bad_epochs, 1)
        self.assertFalse(after_improvement.should_stop)


if __name__ == "__main__":
    unittest.main()
