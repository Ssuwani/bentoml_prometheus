from locust import HttpUser, task
from PIL import Image
import io


class FashionMnistModel(HttpUser):
    @task
    def predict(self):
        image = Image.new("L", (28, 28), 128)  # dummy image
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        # API 요청 보내기
        self.client.post(
            "/predict",
            files={"image": ("images/1.png", image_bytes, "image/png")},
        )
