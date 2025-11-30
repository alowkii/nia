import logging

logging.basicConfig(
    level=logging.INFO,
    filemode="a",
    filename="logs.log",
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    force=True
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)