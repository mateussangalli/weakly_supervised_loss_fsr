import shutil
import unittest

from keras.losses import CategoricalCrossentropy
from keras.models import load_model
from keras.optimizers import Adam

from utils.combined_loss import CombinedLoss
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU
from utils.unet import UNetBuilder
from utils.utils import load_model as load_model_util


class TestLoading(unittest.TestCase):
    def test_without_metrics(self):
        loss_fn = CombinedLoss(
            CategoricalCrossentropy(from_logits=False),
            PRPDirectionalPenalty(3, 2, 5),
            50,
            0.0,
        )

        model = UNetBuilder(
            (None, None, 3),
            8,
            4,
            normalization="batch",
            normalize_all=False,
            batch_norm_momentum=0.85,
        ).build()
        model.compile(
            optimizer=Adam(),
            loss=loss_fn,
        )
        model.save("test_model")

        # create run dir
        custom_objects = {"CombinedLoss": loss_fn}
        model = load_model("test_model", custom_objects=custom_objects, compile=False)

        shutil.rmtree("test_model", ignore_errors=False, onerror=None)

    def test_with_directional_metric(self):
        directional_loss = PRPDirectionalPenalty(3, 2, 5)

        def directional_loss_metric(y, y_pred, **kwargs):
            return directional_loss(y_pred)

        loss_fn = CombinedLoss(
            CategoricalCrossentropy(from_logits=False),
            PRPDirectionalPenalty(3, 2, 5),
            50,
            0.0,
        )

        model = UNetBuilder(
            (None, None, 3),
            8,
            4,
            normalization="batch",
            normalize_all=False,
            batch_norm_momentum=0.85,
        ).build()
        model.compile(optimizer=Adam(), loss=loss_fn, metrics=[directional_loss_metric])
        model.save("test_model")

        custom_objects = {
            "CombinedLoss": loss_fn,
            "directional_loss_metric": directional_loss_metric,
        }
        model = load_model("test_model", custom_objects=custom_objects, compile=False)

        shutil.rmtree("test_model", ignore_errors=False, onerror=None)

    def test_with_iou_metric(self):
        directional_loss = PRPDirectionalPenalty(3, 2, 5)

        def directional_loss_metric(y, y_pred, **kwargs):
            return directional_loss(y_pred)

        loss_fn = CombinedLoss(
            CategoricalCrossentropy(from_logits=False),
            PRPDirectionalPenalty(3, 2, 5),
            50,
            0.0,
        )

        model = UNetBuilder(
            (None, None, 3),
            8,
            4,
            normalization="batch",
            normalize_all=False,
            batch_norm_momentum=0.85,
        ).build()
        model.compile(
            optimizer=Adam(),
            loss=loss_fn,
            metrics=[directional_loss_metric, OneHotMeanIoU(3)],
        )
        model.save("test_model")

        custom_objects = {
            "CombinedLoss": loss_fn,
            "OneHotMeanIoU": OneHotMeanIoU(3),
        }
        model = load_model("test_model", custom_objects=custom_objects, compile=False)

        shutil.rmtree("test_model", ignore_errors=False, onerror=None)

    def test_with_crossentropy(self):
        directional_loss = PRPDirectionalPenalty(3, 2, 5)

        def directional_loss_metric(y, y_pred, **kwargs):
            return directional_loss(y_pred)

        loss_fn = CombinedLoss(
            CategoricalCrossentropy(from_logits=False),
            PRPDirectionalPenalty(3, 2, 5),
            50,
            0.0,
        )

        model = UNetBuilder(
            (None, None, 3),
            8,
            4,
            normalization="batch",
            normalize_all=False,
            batch_norm_momentum=0.85,
        ).build()

        crossentropy = CategoricalCrossentropy(from_logits=False)

        def crossentropy_metric(y_true, y_pred, **kwargs):
            crossentropy(y_true, y_pred)

        model.compile(
            optimizer=Adam(),
            loss=loss_fn,
            metrics=[directional_loss_metric, crossentropy_metric],
        )
        model.save("test_model")

        custom_objects = {
            "CombinedLoss": loss_fn,
            "crossentropy_metric": crossentropy_metric,
        }
        model = load_model("test_model", custom_objects=custom_objects, compile=False)

        shutil.rmtree("test_model", ignore_errors=False, onerror=None)

    def test_func(self):
        directional_loss = PRPDirectionalPenalty(3, 2, 5)

        def directional_loss_metric(y, y_pred, **kwargs):
            return directional_loss(y_pred)

        loss_fn = CombinedLoss(
            CategoricalCrossentropy(from_logits=False),
            PRPDirectionalPenalty(3, 2, 5),
            50,
            0.0,
        )

        model = UNetBuilder(
            (None, None, 3),
            8,
            4,
            normalization="batch",
            normalize_all=False,
            batch_norm_momentum=0.85,
        ).build()

        crossentropy = CategoricalCrossentropy(from_logits=False)

        def crossentropy_metric(y_true, y_pred, **kwargs):
            crossentropy(y_true, y_pred)

        model.compile(
            optimizer=Adam(),
            loss=loss_fn,
            metrics=[directional_loss_metric, crossentropy_metric],
        )
        model.save("test_model")

        model = load_model_util("test_model")

        shutil.rmtree("test_model", ignore_errors=False, onerror=None)


if __name__ == "__main__":
    unittest.main()
