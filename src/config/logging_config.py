import logging

stream_formatter: logging.Formatter = logging.Formatter(
    fmt='[%(name)s] %(levelname)s: %(message)s',
)
file_formatter: logging.Formatter = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

level: int = logging.DEBUG

