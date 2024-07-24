import datetime
import os
from typing import Sequence

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from google.oauth2.service_account import Credentials
from icecream import ic


def get_datetime_now() -> str:
    date = datetime.datetime.now().strftime("%H:%d:%m:%Y")
    return date


# %% Do credential refresh############################
def get_credential(key_path: str) -> Credentials:
    credentials = Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    if credentials.expired:
        ic("Credentials expired. Refreshing...")
        credentials.refresh(Request())
    return credentials


# %% Get credential file############################
def get_key_filepath(key_filepath: str = None, key_dir: str = None) -> str:
    find_json_file = lambda path: [f for f in os.listdir(path) if f.endswith(".json")][
        0
    ]
    if key_filepath is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        # set root to this file dir
        os.chdir(this_file_dir)
        full_path = os.path.join("./keys/", find_json_file("./keys/"))
        return full_path
    else:
        if key_dir is not None:
            full_path = os.path.join(key_dir, find_json_file(key_dir))
            return full_path
        return key_filepath


# %% Get train script############################
def get_trainer_script_filepath(script_filepath: str) -> str:
    if script_filepath is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        # set root to this file dir
        os.chdir(this_file_dir)
        find_py_file = lambda path: [f for f in os.listdir(path) if f.endswith(".py")][
            0
        ]
        full_path = os.path.join("./train_script/", find_py_file("./train_script/"))
        return full_path
    else:
        return script_filepath


class Visual:
    @staticmethod
    def plot_heatmap(data, x_label_list=None, y_label_list=None, title=None):
        fig, ax = plt.subplots(figsize=(50, 3))
        heatmap = ax.pcolor(data, cmap="coolwarm", edgecolors="k", linewidths=0.1)

        # Add color bar to the right of the heatmap
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.remove()

        # Set labels for each axis
        if x_label_list:
            ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
            ax.set_xticklabels(x_label_list, rotation=45, ha="right")
        if y_label_list:
            ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
            ax.set_yticklabels(y_label_list, va="center")

        # Set title
        if title:
            ax.set_title(title)

        plt.tight_layout()

        # Show the plot
        plt.show()

    @staticmethod
    def _plot_2D(
        x_list_np: np.ndarray,
        y_list_np: np.ndarray,
        label_list: Sequence[str],
        title: str,
    ):

        # Create scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(x_list_np, y_list_np, alpha=0.5, edgecolors="k", s=40)

        # Create a mplcursors object to manage the data point interaction
        cursor = mplcursors.cursor(scatter, hover=True)

        # aes
        ax.set_title(title)  # Add a title
        ax.set_xlabel("X_1")  # Add x-axis label
        ax.set_ylabel("X_2")  # Add y-axis label

        # Define how each annotation should look
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(label_list[sel.target.index])
            sel.annotation.get_bbox_patch().set(
                facecolor="white", alpha=0.5
            )  # Set annotation's background color
            sel.annotation.set_fontsize(12)

        plt.show()

    @staticmethod
    def plot_2D(value_list_np: np.ndarray, label_list: Sequence[str], title: str):
        x = value_list_np[:, 0]
        y = value_list_np[:, 1]
        Visual._plot_2D(x, y, label_list, title)

    @staticmethod
    def _plot_clusters_2D(
        x_values,
        y_values,
        label_list,
        cluster_labels_or_centroid_ids,
        title: str,
    ):
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            x_values,
            y_values,
            c=cluster_labels_or_centroid_ids,
            cmap="Set1",
            alpha=0.5,
            edgecolors="k",
            s=40,
        )  # Change the denominator as per n_clusters

        # Create a mplcursors object to manage the data point interaction
        cursor = mplcursors.cursor(scatter, hover=True)

        # axes
        ax.set_title(title)  # Add a title
        ax.set_xlabel("X_1")  # Add x-axis label
        ax.set_ylabel("X_2")  # Add y-axis label

        # Define how each annotation should look
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(label_list.category[sel.target.index])
            sel.annotation.get_bbox_patch().set(
                facecolor="white", alpha=0.95
            )  # Set annotation's background color
            sel.annotation.set_fontsize(14)

        plt.show()

    @staticmethod
    def plot_clusters_2D(
        value_list_np: np.ndarray,
        label_list: Sequence[str],
        cluster_labels_or_centroid_ids: np.ndarray,
        title: str,
    ):
        x = value_list_np[:, 0]
        y = value_list_np[:, 1]
        Visual._plot_clusters_2D(
            x, y, label_list, cluster_labels_or_centroid_ids, title
        )
