from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class UploadLimitMiddleware:
    """Reject oversized bodies before Starlette finishes spooling multipart files."""

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.limit:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length."}, 400)(scope, receive, send)
        if length > self.limit:
            return await JSONResponse({"detail": "The upload is too large."}, 413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.limit:
                raise StarletteHTTPException(413, "The upload is too large.")
            return message

        await self.app(scope, limited_receive, send)
