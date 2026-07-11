"""Gemeinsame Hilfsfunktionen für das Federated-Learning-Demoprojekt.

Dieses Modul wird sowohl vom Server-Notebook (``server.ipynb``) als auch von
jedem Client-Notebook (``client.ipynb``) importiert. Es enthält:

* die Definition des gemeinsam trainierten Keras-Modells (:func:`create_model`),
* Funktionen zur reproduzierbaren Partitionierung des MNIST-Datensatzes auf
  20 Clients, sowohl IID (:func:`load_partition`) als auch Non-IID
  (:func:`load_partition_shard` und :func:`load_partition_dominant_class`),
  jeweils erreichbar über die zentrale Konfiguration in ``FLConfig``,
* Normalisierungs- und Visualisierungshilfen.

Alle Zufallsprozesse sind über einen festen Seed reproduzierbar, damit jede
Studierendengruppe bei gleicher ``CLIENT_ID`` exakt dieselbe Partition erhält.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Zentrale Konfiguration
# ---------------------------------------------------------------------------

#: Fester Seed für alle Zufallsoperationen (Mischen, Shard-Zuweisung, ...).
#: WICHTIG: Server und alle Clients müssen denselben Seed verwenden, damit
#: die Partitionierung reproduzierbar und für alle identisch ist.
RANDOM_SEED = 42

#: Anzahl der Clients (= Anzahl der Studierenden). Zentral konfigurierbar.
NUM_CLIENTS = 20

#: Art der Datenverteilung auf die Clients.
#:   - "iid": Jeder Client bekommt einen zufälligen, gleich großen Anteil
#:            aller Klassen (Standard für die erste Ausbaustufe).
#:   - "shard": Non-IID nach dem ursprünglichen FedAvg-Paper (McMahan et al.,
#:              2017). Jeder Client bekommt überwiegend zwei Klassen.
#:   - "dominant": Non-IID-Variante, bei der jeder Client zu ca. 80-90 % aus
#:                 einer dominanten Klasse besteht und zu kleinen Teilen aus
#:                 allen übrigen Klassen.
PartitionStrategy = Literal["iid", "shard", "dominant"]

DEFAULT_PARTITION_STRATEGY: PartitionStrategy = "iid"


@dataclass
class FLConfig:
    """Zentrale, veränderbare Konfiguration für Server und Clients.

    Alle Stellschrauben des Experiments sind hier gebündelt, damit Dozierende
    und Studierende nicht an mehreren Stellen im Code suchen müssen.

    Attributes:
        num_clients: Anzahl der Clients, auf die MNIST aufgeteilt wird.
        num_rounds: Anzahl der Federated-Learning-Kommunikationsrunden.
        local_epochs: Anzahl lokaler Trainings-Epochen pro Runde und Client.
        fraction_fit: Anteil der verfügbaren Clients, die pro Runde am
            Training teilnehmen (z. B. 0.5 = nur die Hälfte trainiert).
        fraction_evaluate: Anteil der verfügbaren Clients, die pro Runde an
            der föderierten Evaluation teilnehmen.
        min_fit_clients: Mindestanzahl an Clients, die für eine Trainingsrunde
            zur Verfügung stehen müssen.
        min_available_clients: Mindestanzahl an Clients, die insgesamt mit
            dem Server verbunden sein müssen, bevor eine Runde startet.
        partition_strategy: Siehe :data:`PartitionStrategy`.
        batch_size: Batchgröße für das lokale Training.
        random_seed: Seed für alle reproduzierbaren Zufallsoperationen.
    """

    num_clients: int = NUM_CLIENTS
    num_rounds: int = 10
    local_epochs: int = 1
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_available_clients: int = 2
    partition_strategy: PartitionStrategy = DEFAULT_PARTITION_STRATEGY
    batch_size: int = 32
    random_seed: int = RANDOM_SEED


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------

def create_model() -> tf.keras.Model:
    """Erzeugt das gemeinsame Keras-Modell für Server und Clients.

    Alle Teilnehmenden (Server zur Initialisierung, jeder Client lokal)
    müssen exakt dieselbe Architektur verwenden, da sonst die Gewichte beim
    Federated Averaging nicht kompatibel sind.

    Returns:
        tf.keras.Model: Ein kompiliertes, einfaches Feed-Forward-Netz mit
        einer Flatten-Schicht, einer versteckten Dense-Schicht (ReLU) und
        einer Softmax-Ausgabeschicht für die 10 MNIST-Klassen.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------------

def normalize(images: np.ndarray) -> np.ndarray:
    """Normiert Bildpixelwerte von [0, 255] auf [0, 1].

    Args:
        images: Array von Graustufenbildern, z. B. der Form
            ``(n, 28, 28)`` mit Werten im Bereich [0, 255].

    Returns:
        np.ndarray: Array derselben Form mit ``float32``-Werten im
        Bereich [0, 1].
    """
    return images.astype("float32") / 255.0


# ---------------------------------------------------------------------------
# IID-Partitionierung
# ---------------------------------------------------------------------------

def _load_mnist_shuffled(seed: int = RANDOM_SEED):
    """Lädt MNIST und mischt die Trainingsdaten reproduzierbar.

    Args:
        seed: Seed für den reproduzierbaren Zufallsgenerator.

    Returns:
        tuple: ``(x_train, y_train, x_test, y_test)`` bereits gemischte
        Trainingsdaten (Testdaten bleiben unverändert für die globale
        Evaluation auf dem Server).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_train))
    return x_train[perm], y_train[perm], x_test, y_test


def load_partition(
    client_id: int,
    num_clients: int = NUM_CLIENTS,
    seed: int = RANDOM_SEED,
):
    """Lädt den IID-Datenanteil (gleich großer, zufälliger Block) eines Clients.

    Die 60.000 MNIST-Trainingsbilder werden reproduzierbar zufällig gemischt
    und anschließend in ``num_clients`` gleich große, disjunkte Blöcke
    aufgeteilt. Client ``i`` erhält Block ``i``. Da alle Clients denselben
    Seed verwenden, ist die Partitionierung für alle deterministisch und
    wiederholbar.

    Args:
        client_id: Index des Clients (0-basiert, z. B. 0 bis 19).
        num_clients: Gesamtanzahl der Clients, auf die aufgeteilt wird.
        seed: Seed für das reproduzierbare Mischen der Daten.

    Returns:
        tuple: ``(x_partition, y_partition)`` — die (bereits normierten)
        Bilder und Labels für diesen Client.

    Raises:
        ValueError: Wenn ``client_id`` außerhalb des gültigen Bereichs liegt.
    """
    if not 0 <= client_id < num_clients:
        raise ValueError(
            f"client_id muss zwischen 0 und {num_clients - 1} liegen, "
            f"erhalten: {client_id}"
        )

    x_train, y_train, _, _ = _load_mnist_shuffled(seed)
    block_size = len(x_train) // num_clients
    start = client_id * block_size
    end = start + block_size

    x_partition = normalize(x_train[start:end])
    y_partition = y_train[start:end]
    return x_partition, y_partition


# ---------------------------------------------------------------------------
# Non-IID-Partitionierung: Variante 1 - Shard-Methode (FedAvg-Paper)
# ---------------------------------------------------------------------------

def load_partition_shard(
    client_id: int,
    num_clients: int = NUM_CLIENTS,
    shards_per_client: int = 2,
    seed: int = RANDOM_SEED,
):
    """Lädt eine Non-IID-Partition nach der Shard-Methode (McMahan et al., 2017).

    Die Trainingsdaten werden zunächst nach Klassenlabel sortiert und in
    ``num_clients * shards_per_client`` gleich große "Shards" zerlegt. Jeder
    Shard enthält somit überwiegend Bilder einer einzigen Klasse (an den
    Rändern können durch die Sortierung zwei Klassen in einem Shard liegen).
    Anschließend werden die Shards reproduzierbar zufällig gemischt und
    jedem Client ``shards_per_client`` Shards zugewiesen. Mit dem
    Standardwert ``shards_per_client=2`` besitzt jeder Client somit
    hauptsächlich zwei Klassen — exakt wie im ursprünglichen FedAvg-Paper.

    Args:
        client_id: Index des Clients (0-basiert).
        num_clients: Gesamtanzahl der Clients.
        shards_per_client: Anzahl der Shards, die jeder Client erhält.
        seed: Seed für die reproduzierbare Shard-Zuweisung.

    Returns:
        tuple: ``(x_partition, y_partition)`` — die (normierten) Bilder und
        Labels für diesen Client.

    Raises:
        ValueError: Wenn ``client_id`` außerhalb des gültigen Bereichs liegt.
    """
    if not 0 <= client_id < num_clients:
        raise ValueError(
            f"client_id muss zwischen 0 und {num_clients - 1} liegen, "
            f"erhalten: {client_id}"
        )

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Nach Label sortieren, damit jeder Shard möglichst homogen ist.
    sort_idx = np.argsort(y_train, kind="stable")
    x_sorted, y_sorted = x_train[sort_idx], y_train[sort_idx]

    num_shards = num_clients * shards_per_client
    shard_size = len(x_sorted) // num_shards
    shard_indices = np.arange(num_shards)

    rng = np.random.default_rng(seed)
    rng.shuffle(shard_indices)

    assigned_shards = shard_indices[
        client_id * shards_per_client : (client_id + 1) * shards_per_client
    ]

    x_parts, y_parts = [], []
    for shard_id in assigned_shards:
        start = shard_id * shard_size
        end = start + shard_size
        x_parts.append(x_sorted[start:end])
        y_parts.append(y_sorted[start:end])

    x_partition = np.concatenate(x_parts)
    y_partition = np.concatenate(y_parts)

    # Innerhalb der Partition nochmals mischen, damit Trainings-Batches nicht
    # blockweise nach Shard sortiert sind.
    perm = rng.permutation(len(x_partition))
    return normalize(x_partition[perm]), y_partition[perm]


# ---------------------------------------------------------------------------
# Non-IID-Partitionierung: Variante 2 - Dominante Klasse
# ---------------------------------------------------------------------------

def load_partition_dominant_class(
    client_id: int,
    num_clients: int = NUM_CLIENTS,
    dominant_fraction: float = 0.85,
    seed: int = RANDOM_SEED,
):
    """Lädt eine Non-IID-Partition mit einer dominanten Klasse pro Client.

    Jeder Client erhält ``dominant_fraction`` (Standard: 85 %) seiner Daten
    aus einer "dominanten" Klasse (``client_id % 10``, sodass bei 20 Clients
    jede der 10 Ziffernklassen genau zwei dominante Clients hat) sowie den
    Rest gleichmäßig verteilt über alle übrigen Klassen.

    Args:
        client_id: Index des Clients (0-basiert). Die dominante Klasse
            ergibt sich aus ``client_id % 10``.
        num_clients: Gesamtanzahl der Clients.
        dominant_fraction: Anteil der Bilder aus der dominanten Klasse
            (zwischen 0 und 1).
        seed: Seed für die reproduzierbare Stichprobenziehung.

    Returns:
        tuple: ``(x_partition, y_partition)`` — die (normierten) Bilder und
        Labels für diesen Client.

    Raises:
        ValueError: Wenn ``client_id`` außerhalb des gültigen Bereichs liegt
            oder ``dominant_fraction`` nicht zwischen 0 und 1 liegt.
    """
    if not 0 <= client_id < num_clients:
        raise ValueError(
            f"client_id muss zwischen 0 und {num_clients - 1} liegen, "
            f"erhalten: {client_id}"
        )
    if not 0.0 < dominant_fraction < 1.0:
        raise ValueError("dominant_fraction muss zwischen 0 und 1 liegen")

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    dominant_class = client_id % 10
    samples_per_client = len(x_train) // num_clients
    n_dominant = int(samples_per_client * dominant_fraction)
    n_rest = samples_per_client - n_dominant

    rng = np.random.default_rng(seed + client_id)  # client-spezifisch, aber reproduzierbar

    dominant_idx = np.where(y_train == dominant_class)[0]
    other_idx = np.where(y_train != dominant_class)[0]

    chosen_dominant = rng.choice(dominant_idx, size=n_dominant, replace=False)
    chosen_rest = rng.choice(other_idx, size=n_rest, replace=False)

    all_idx = np.concatenate([chosen_dominant, chosen_rest])
    rng.shuffle(all_idx)

    return normalize(x_train[all_idx]), y_train[all_idx]


def load_test_data():
    """Lädt den globalen MNIST-Testdatensatz (normiert) für die Evaluation.

    Der Testdatensatz wird nicht auf die Clients aufgeteilt, sondern von
    Server und Clients gleichermaßen zur (globalen bzw. lokalen) Evaluation
    des Modells verwendet.

    Returns:
        tuple: ``(x_test, y_test)`` — normierte Testbilder und zugehörige
        Labels.
    """
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return normalize(x_test), y_test


def load_client_data(
    client_id: int,
    strategy: PartitionStrategy = DEFAULT_PARTITION_STRATEGY,
    num_clients: int = NUM_CLIENTS,
    seed: int = RANDOM_SEED,
):
    """Einheitlicher Einstiegspunkt zum Laden der Client-Daten nach Strategie.

    Kapselt :func:`load_partition`, :func:`load_partition_shard` und
    :func:`load_partition_dominant_class` hinter einer zentral im
    Client-Notebook konfigurierbaren Auswahl.

    Args:
        client_id: Index des Clients (0-basiert).
        strategy: Eine der Partitionierungsstrategien aus
            :data:`PartitionStrategy` ("iid", "shard" oder "dominant").
        num_clients: Gesamtanzahl der Clients.
        seed: Seed für die reproduzierbare Partitionierung.

    Returns:
        tuple: ``(x_partition, y_partition)`` für diesen Client.

    Raises:
        ValueError: Wenn ``strategy`` unbekannt ist.
    """
    if strategy == "iid":
        return load_partition(client_id, num_clients=num_clients, seed=seed)
    if strategy == "shard":
        return load_partition_shard(client_id, num_clients=num_clients, seed=seed)
    if strategy == "dominant":
        return load_partition_dominant_class(
            client_id, num_clients=num_clients, seed=seed
        )
    raise ValueError(
        f"Unbekannte Partitionierungsstrategie: {strategy!r}. "
        "Erlaubt sind 'iid', 'shard', 'dominant'."
    )


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def plot_distribution(
    y_partition: np.ndarray,
    client_id: int | None = None,
    ax: plt.Axes | None = None,
):
    """Visualisiert die Klassenverteilung einer Datenpartition als Histogramm.

    Args:
        y_partition: Array der Labels (Ziffern 0-9) einer Client-Partition.
        client_id: Optionale Client-ID, die im Titel angezeigt wird.
        ax: Optionale Matplotlib-Achse, auf der gezeichnet werden soll. Wird
            keine übergeben, erzeugt die Funktion eine neue Figure.

    Returns:
        matplotlib.axes.Axes: Die Achse mit dem gezeichneten Histogramm.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    counts = np.bincount(y_partition, minlength=10)
    ax.bar(range(10), counts, color="#3b6ea5")
    ax.set_xticks(range(10))
    ax.set_xlabel("Ziffer (Klasse)")
    ax.set_ylabel("Anzahl Bilder")
    title = "Klassenverteilung"
    if client_id is not None:
        title += f" – Client {client_id}"
    ax.set_title(title)
    return ax


def plot_accuracy_over_rounds(rounds: list[int], accuracies: list[float]):
    """Zeichnet die globale Accuracy über die Kommunikationsrunden.

    Args:
        rounds: Liste der Rundennummern (z. B. ``[1, 2, 3, ...]``).
        accuracies: Liste der zugehörigen globalen Accuracy-Werte.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure mit dem Liniendiagramm.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, accuracies, marker="o", color="#3b6ea5", linewidth=2)
    ax.set_xlabel("Kommunikationsrunde")
    ax.set_ylabel("Globale Accuracy")
    ax.set_title("Globale Accuracy über Kommunikationsrunden")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    return fig


def plot_local_training_curve(
    epochs: list[int],
    accuracies: list[float],
    federated_accuracy: float | None = None,
):
    """Zeichnet die Test-Accuracy eines rein lokal trainierten Modells über die Epochen.

    Wird im Baseline-Notebook (`local_baseline.ipynb`) verwendet, um zu zeigen,
    wie sich ein Modell entwickelt, das ausschließlich auf der eigenen
    Client-Partition trainiert wird — ohne Federated Averaging. Optional wird
    zusätzlich eine horizontale Referenzlinie mit der zuvor erreichten
    federated Accuracy eingezeichnet, um den direkten Vergleich zu erleichtern.

    Args:
        epochs: Liste der Epochennummern (z. B. ``[1, 2, 3, ...]``).
        accuracies: Liste der zugehörigen Test-Accuracy-Werte nach jeder Epoche.
        federated_accuracy: Optionale, zuvor im Server-Notebook erreichte
            finale globale Accuracy, die als gestrichelte Referenzlinie
            eingezeichnet wird. Wird ``None`` übergeben, entfällt die Linie.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure mit dem Liniendiagramm.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        epochs, accuracies, marker="o", color="#c0392b", linewidth=2,
        label="Nur lokales Training (dieser Client)",
    )
    if federated_accuracy is not None:
        ax.axhline(
            federated_accuracy, color="#3b6ea5", linestyle="--", linewidth=2,
            label=f"Federated Learning (final: {federated_accuracy:.4f})",
        )
    ax.set_xlabel("Lokale Trainings-Epoche")
    ax.set_ylabel("Test-Accuracy (globaler Testdatensatz)")
    ax.set_title("Nur lokales Training vs. Federated Learning")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return fig


def plot_local_vs_federated_bar(local_accuracy: float, federated_accuracy: float):
    """Zeichnet ein einfaches Balkendiagramm: lokale vs. federated Accuracy.

    Args:
        local_accuracy: Finale Test-Accuracy des rein lokal trainierten Modells.
        federated_accuracy: Finale globale Test-Accuracy des federated
            trainierten Modells (vom Server-Notebook mitgeteilt).

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure mit dem Balkendiagramm.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["Nur lokales\nTraining", "Federated\nLearning"]
    values = [local_accuracy, federated_accuracy]
    colors = ["#c0392b", "#3b6ea5"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Test-Accuracy (globaler Testdatensatz)")
    ax.set_ylim(0, 1)
    ax.set_title("Vergleich der finalen Test-Accuracy")
    ax.grid(alpha=0.3, axis="y")
    return fig
