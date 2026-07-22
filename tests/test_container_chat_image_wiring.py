from app.core.container import Container
from app.repositories.chat_image import ChatImageRepository
from app.services.chat_image_service import ChatImageStorageService


def test_container_provides_chat_image_components():
    c = Container()
    assert isinstance(c.chat_image_repository(), ChatImageRepository)
    assert isinstance(c.chat_image_service(), ChatImageStorageService)
