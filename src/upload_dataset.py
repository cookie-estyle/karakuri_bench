import weave
from weave import Dataset
import polars as pl

weave.init("karakuri-bench/weave-test")

df = pl.read_csv('data/tasks/questions.csv')
rows = df.to_dicts()
dataset = Dataset(
    name='karakuri-bench-dataset',
    rows=rows
)
weave.publish(dataset)