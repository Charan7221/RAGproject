"""
Request Logging Middleware
Captures all HTTP requests/responses with detailed information
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
import time
import json
from typing import Callable
from .logging_config import get_logger, set_session_context

logger = get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests and outgoing responses.
    Extracts session_id from request and adds to logging context.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Start timer
        start_time = time.time()
        
        # Extract session_id from various sources
        session_id = await self._extract_session_id(request)
        
        # Set session context for logging
        set_session_context(session_id)
        
        # Generate request ID for tracking
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        # Log incoming request
        logger.info(
            f"REQUEST [{request_id}] {request.method} {request.url.path} | "
            f"Client: {request.client.host}:{request.client.port} | "
            f"Session: {session_id or 'N/A'}"
        )
        
        # Log query parameters if present
        if request.query_params:
            logger.debug(f"Query Params [{request_id}]: {dict(request.query_params)}")
        
        # Log request body for POST/PUT requests (be careful with large files)
        if request.method in ["POST", "PUT", "PATCH"]:
            await self._log_request_body(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"RESPONSE [{request_id}] {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Time: {process_time:.3f}s"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"ERROR [{request_id}] {request.method} {request.url.path} | "
                f"Time: {process_time:.3f}s | "
                f"Error: {str(e)}"
            )
            raise
    
    async def _extract_session_id(self, request: Request) -> str:
        """Extract session_id from request (query params, body, or headers)."""
        
        # Try query parameters first
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id
        
        # Try to extract from request body (for JSON requests)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Check content type
                content_type = request.headers.get("content-type", "")
                
                if "application/json" in content_type:
                    # Read body (we need to make it re-readable)
                    body = await request.body()
                    if body:
                        try:
                            body_json = json.loads(body)
                            session_id = body_json.get("session_id")
                            if session_id:
                                return session_id
                        except json.JSONDecodeError:
                            pass
                    
                    # Body is cached by Starlette in request._body, no need to mock receive
                    pass
                    
            except Exception as e:
                logger.debug(f"Could not extract session_id from body: {e}")
        
        # Try headers
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return session_id
        
        return None
    
    async def _log_request_body(self, request: Request, request_id: str):
        """Log request body (with size limits for safety)."""
        try:
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                body = await request.body()
                if body:
                    # Limit log size
                    if len(body) < 1000:  # Only log small payloads
                        try:
                            body_json = json.loads(body)
                            # Redact sensitive fields
                            safe_body = self._redact_sensitive_fields(body_json)
                            logger.debug(f"Request Body [{request_id}]: {safe_body}")
                        except json.JSONDecodeError:
                            logger.debug(f"Request Body [{request_id}]: <non-JSON data>")
                    else:
                        logger.debug(f"Request Body [{request_id}]: <large payload, {len(body)} bytes>")
                    
                    # Body is cached by Starlette in request._body, no need to mock receive
                    pass
                    
            elif "multipart/form-data" in content_type:
                logger.debug(f"Request Body [{request_id}]: <file upload>")
                
        except Exception as e:
            logger.debug(f"Could not log request body: {e}")
    
    def _redact_sensitive_fields(self, data: dict) -> dict:
        """Redact sensitive fields from logged data."""
        sensitive_keys = ['password', 'api_key', 'token', 'secret', 'credential']
        
        if not isinstance(data, dict):
            return data
        
        redacted = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                redacted[key] = "***REDACTED***"
            elif isinstance(value, dict):
                redacted[key] = self._redact_sensitive_fields(value)
            else:
                redacted[key] = value
        
        return redacted
