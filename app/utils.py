# app/utils.py
import base64
import io

import matplotlib.pyplot as plt

__all__ = ["fig_to_base64"]


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
