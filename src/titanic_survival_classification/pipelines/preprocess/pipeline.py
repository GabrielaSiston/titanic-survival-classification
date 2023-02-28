"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.5
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data, predict

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=preprocess_data,
            inputs=['train', 'test'],
            outputs=['train_num', 'test_num'],
            name='preprocess_data_node'
        ),
        node(
            func=predict,
            inputs=['train_num', 'test_num'],
            #outputs=['result'],
            outputs='result',
            name='result'
        )
    ])

