import bentoml
from bentoml.io import Image as PILImage
import numpy as np

bento_model = bentoml.pytorch.get("fashion_mnist_model:latest")
runner = bento_model.to_runner()

svc = bentoml.Service(
    name="fashion_mnist_model",
    runners=[runner],
)

labels = bento_model.custom_objects["labels"]


@svc.api(input=bentoml.io.Image(), output=bentoml.io.JSON())
async def predict(image: PILImage):
    arr = np.array(image)
    assert arr.shape == (28, 28)

    arr = np.expand_dims(arr, 0).astype("float32")  # 차원 추가
    output_tensor = await runner.async_run(arr)
    prediction = output_tensor.numpy().argmax()

    return {"label": labels[prediction], "idx": prediction}
