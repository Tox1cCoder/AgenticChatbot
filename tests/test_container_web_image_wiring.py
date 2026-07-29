from app.core.container import Container
from app.repositories.web_image_reference import WebImageReferenceRepository
from app.services.web_image_service import WebImageService


def test_container_provides_async_web_image_components():
    container = Container()

    repository = container.web_image_reference_repository()
    assert isinstance(repository, WebImageReferenceRepository)
    assert repository.async_session_factory is not None
    assert isinstance(container.web_image_service(), WebImageService)
