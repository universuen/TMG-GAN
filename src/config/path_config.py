from pathlib import Path

src: Path = Path(__file__).absolute().parent.parent
project: Path = src.parent
data: Path = project / 'data'
logs: Path = data / 'logs'
datasets: Path = data / 'datasets'
gan_outs: Path = data / 'gan_outs'

for i in list(vars().values()):
    if isinstance(i, Path):
        i.mkdir(parents=True, exist_ok=True)
