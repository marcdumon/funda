from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import altair as alt


def plot_confusion_matrix(name: str, y_true: np.array, y_pred: np.array, normalize=True):  # Todo: normalize to [0..1]
    cm = confusion_matrix(y_true, y_pred)

    labels = list((set(y_true.astype(np.int).astype(np.str))))  # Todo: what if the labels are already strings?

    cm = pd.DataFrame(cm, columns=labels)
    df = pd.DataFrame(list(product(cm.index, repeat=2)), columns=['Actual', 'Predicted'])
    df['results'] = cm.values.flatten()

    chart = alt.Chart(df, height=250, width=250).mark_rect().encode(x='Actual:O', y='Predicted:O', color='results:Q')

    # Configure text
    text = chart.mark_text(baseline='middle').encode(
        text='results:Q',
        color=alt.condition(
            alt.datum.results < 500,  # Todo: make generic
            alt.value('black'),
            alt.value('white')
        ))
    heatmap = chart + text
    # chart.save('xxx.html')
    heatmap.save(name, format='png', scale_factor=2.0)
