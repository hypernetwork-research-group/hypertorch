from urllib.parse import urlparse


def validate_http_url(value: str) -> str:
    """
    Validate that a URL uses HTTP or HTTPS.

    Args:
        value: URL string to validate.

    Returns:
        value: The validated URL.

    Raises:
        ValueError: If the URL is not an absolute HTTP or HTTPS URL.
    """
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    return value
