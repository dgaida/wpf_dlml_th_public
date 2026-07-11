# Federated Learning mit Flower AI in Google Colab

Ein kleines Open-Source-Lehrprojekt zur Demonstration von **Federated Learning** in einer
Lehrveranstaltung: Eine Dozentin / ein Dozent startet einen Flower-Server in Google Colab, bis zu
20 Studierende verbinden sich mit jeweils eigenem Colab-Notebook als Clients, und gemeinsam wird
per **Federated Averaging (FedAvg)** ein neuronales Netz auf dem MNIST-Datensatz trainiert — ohne
dass jemals Trainingsdaten den eigenen Rechner verlassen.

## Inhalt

```
fl-colab-demo/
├── server.ipynb           # Vom Dozenten/von der Dozentin einmal ausgeführt (Flower-Server)
├── client.ipynb           # Von jeder/jedem Studierenden ausgeführt (Flower-Client)
├── local_baseline.ipynb   # Optional: Vergleich "nur lokales Training" vs. Federated Learning
├── common.py              # Gemeinsame Hilfsfunktionen (Modell, Datenpartitionierung, Plots)
├── slides.pdf              # Foliensatz für die Veranstaltung
├── figures/                # Von den Notebooks erzeugte Diagramme
└── README.md               # Diese Datei
```

## Architektur

```
                     Google Colab (Dozent:in)
                        Flower Server (FedAvg)
                                |
                          ngrok-Tunnel
                                |
      ------------------------------------------------------
      |        |        |        |       ...       |
     C0       C1       C2       C3               C19
                Google Colab Clients (Studierende)
```

Da Google Colab von außen nicht direkt erreichbar ist, macht der Server sich über einen
**ngrok-Tunnel** öffentlich erreichbar. Die Studierenden verbinden sich anschließend direkt mit
dieser Adresse — kein zusätzlicher Infrastruktur-Aufwand nötig.

> **Hinweis zur Flower-API:** Aktuelle Flower-Versionen empfehlen für produktive Deployments das
> neue SuperLink/SuperNode-Runtime (`flwr run`). Dieses Runtime ist auf dauerhaft betriebene
> Infrastruktur mit separat gestarteten Prozessen ausgelegt und passt nicht zu 20 unabhängig
> gestarteten Colab-Notebooks in einer einzelnen Sitzung. Dieses Projekt verwendet daher bewusst
> die weiterhin unterstützte, adressbasierte `start_server()`/`start_client()`-API (auch als
> *Compat*-API bezeichnet). Die dabei erscheinenden Deprecation-Hinweise in den Logs sind für
> dieses Lehrszenario unproblematisch und können ignoriert werden.

## Voraussetzungen

- Ein Google-Konto (für Google Colab) für Dozent:in und alle Studierenden.  
- Ein kostenloser [ngrok-Account](https://dashboard.ngrok.com/signup) samt Authtoken — **nur für  
  die Dozentin / den Dozenten** nötig, um den Server-Tunnel zu öffnen.  
- Keine lokale Installation notwendig; alle Abhängigkeiten werden in Colab installiert.  

## Anleitung für den Dozenten / die Dozentin

1. **`server.ipynb` öffnen** (z. B. per Klick auf die Datei in GitHub → "Open in Colab", oder  
   Upload in Google Colab).  
2. In Abschnitt 5 den eigenen **ngrok-Authtoken** hinterlegen (am einfachsten als Colab-Secret  
   `NGROK_AUTH_TOKEN` über das 🔑-Symbol in der linken Seitenleiste).  
3. In Abschnitt 3 die **zentrale Konfiguration** an die Veranstaltung anpassen, insbesondere  
   `min_available_clients` und `min_fit_clients` (z. B. auf die Hälfte der erwarteten
   Studierendenzahl setzen, damit der Server nicht auf einzelne fehlende Clients wartet).  
4. **Notebook Zelle für Zelle ausführen.** Nach Abschnitt 5 (ngrok-Tunnel) wird eine Adresse  
   ausgegeben, z. B.:
   ```
   0.tcp.eu.ngrok.io:12345
   ```
5. **Diese Adresse den Studierenden mitteilen** (Chat, Whiteboard, Foliensatz).  
6. Die letzte Zelle in Abschnitt 8 startet den Server und **blockiert**, bis alle konfigurierten  
   Kommunikationsrunden abgeschlossen sind oder genug Clients verbunden sind. Der Fortschritt wird
   nach jeder Runde als `Global Loss` / `Global Accuracy` ausgegeben.  
7. Nach Abschluss zeigen die letzten Zellen die Accuracy-Kurve über die Runden und speichern das  
   finale globale Modell (`global_model.keras`).

## Anleitung für Studierende

1. **`client.ipynb` öffnen** in Google Colab.  
2. In Abschnitt 3 die **beiden** vom Dozenten / von der Dozentin erhaltenen Werte eintragen:  
   ```python
   SERVER_ADDRESS = "0.tcp.eu.ngrok.io:12345"  # von Dozent:in erhalten
   CLIENT_ID = 7                                # eigene, zugewiesene Nummer (0-19)
   ```
3. **"Laufzeit → Alle Zellen ausführen"** wählen.  
4. Das Notebook lädt automatisch den eigenen Datenanteil (Abschnitt 4), zeigt optional die eigene  
   Klassenverteilung (Abschnitt 5) und verbindet sich anschließend mit dem Server (Abschnitt 8).  
5. Die letzte Zelle **blockiert**, während im Hintergrund abwechselnd lokal trainiert und  
   evaluiert wird. Die Konsolenausgabe zeigt die lokale Trainings- und Evaluations-Accuracy pro
   Runde.  
6. Sobald der Server alle Runden abgeschlossen hat, endet die Zelle automatisch — fertig!  

## Anleitung für `local_baseline.ipynb` (optional)

Dieses dritte Notebook braucht **keine Verbindung zum Server** und kann jederzeit unabhängig
ausgeführt werden — auch nachträglich zu Hause. Es lässt Studierende direkt erleben, warum
Federated Learning einen Mehrwert bringt:

1. `local_baseline.ipynb` öffnen.  
2. In Abschnitt 3 dieselbe `CLIENT_ID` und `PARTITION_STRATEGY` wie in `client.ipynb` eintragen,  
   sowie optional die am Ende von `server.ipynb` ausgegebene finale `Global Accuracy` als
   `FEDERATED_ACCURACY`.  
3. Alle Zellen ausführen. Das Notebook trainiert ein Modell **ausschließlich** auf der eigenen  
   Datenpartition (gleiche Gesamt-Epochenzahl wie im federated Training, für einen fairen
   Vergleich) und stellt die erreichte Accuracy der federated erreichten Accuracy gegenüber.  
4. Besonders bei `partition_strategy = "shard"` oder `"dominant"` (Non-IID) wird der Unterschied  
   deutlich sichtbar: Das rein lokale Modell kennt nur wenige Ziffern gut, das federated Modell
   profitiert vom Wissen aller Clients.

## Zentrale Konfiguration

Alle Experiment-Parameter sind in `common.py` in der Klasse `FLConfig` gebündelt und müssen nur an
**einer Stelle** (im Server-Notebook, Abschnitt 3) angepasst werden:

| Parameter | Bedeutung |
|---|---|
| `num_clients` | Anzahl der Clients, auf die MNIST aufgeteilt wird (Standard: 20) |
| `num_rounds` | Anzahl der Kommunikationsrunden |
| `local_epochs` | Anzahl lokaler Trainings-Epochen pro Runde und Client |
| `fraction_fit` | Anteil der verfügbaren Clients, die pro Runde trainieren |
| `fraction_evaluate` | Anteil der verfügbaren Clients, die pro Runde evaluieren |
| `min_fit_clients` | Mindestanzahl an Clients für eine Trainingsrunde |
| `min_available_clients` | Mindestanzahl insgesamt verbundener Clients, bevor eine Runde startet |
| `partition_strategy` | `"iid"`, `"shard"` oder `"dominant"` (siehe unten) |
| `batch_size` | Batchgröße für das lokale Training |
| `random_seed` | Seed für alle reproduzierbaren Zufallsoperationen |

`partition_strategy` muss serverseitig **und** in jedem Client-Notebook identisch gesetzt werden,
da sonst inkonsistente Datenverteilungen entstehen.

## IID vs. Non-IID: Datenverteilungen

Diese Demo unterstützt drei Partitionierungsstrategien (implementiert in `common.py`):

- **`"iid"`** (Standard): Die 60.000 MNIST-Trainingsbilder werden reproduzierbar gemischt und in  
  `num_clients` gleich große, zufällige Blöcke aufgeteilt. Jeder Client sieht somit eine
  repräsentative Mischung aller zehn Ziffernklassen.  
- **`"shard"`**: Non-IID-Partitionierung nach der Shard-Methode aus dem ursprünglichen  
  FedAvg-Paper (McMahan et al., 2017). Die Daten werden nach Klasse sortiert, in kleine Shards
  zerlegt, und jeder Client erhält zwei zufällige Shards — er besitzt somit überwiegend nur zwei
  Ziffernklassen.  
- **`"dominant"`**: Jeder Client erhält ca. 80–90 % seiner Daten aus einer dominanten Klasse  
  (`client_id % 10`) und den Rest gleichmäßig über alle übrigen Klassen verteilt.

Die optionale Visualisierung in Abschnitt 5 des Client-Notebooks (`common.plot_distribution`)
macht den Unterschied zwischen den Strategien unmittelbar sichtbar.

## Erweiterungen

Das Projekt ist bewusst leicht erweiterbar gehalten:

- **Nur ein Teil der Clients trainiert pro Runde:** `fraction_fit` auf z. B. `0.5` setzen.  
- **Client-Ausfälle:** Schließt eine/ein Studierende:r ihr/sein Notebook, arbeitet der Server  
  weiter, solange `min_available_clients` erfüllt bleibt.  
- **Unterschiedliche Trainingsdauer:** In `fit_config()` im Server-Notebook lässt sich  
  `local_epochs` z. B. abhängig von der Runde oder pro Client variieren.  
- **Zentrales Training zum Vergleich:** `common.create_model()` einmal auf dem vollständigen,  
  unpartitionierten MNIST-Trainingsdatensatz trainieren und die Accuracy der federated erreichten
  Accuracy gegenüberstellen.

## Fehlerbehebung

- **Server startet nicht / keine ngrok-Adresse:** Authtoken prüfen (Abschnitt 5 im  
  Server-Notebook); ohne gültigen Token verweigert ngrok den Tunnel.  
- **Client verbindet nicht:** `SERVER_ADDRESS` exakt wie ausgegeben übernehmen (ohne `tcp://`-  
  Präfix, das übernimmt das Notebook automatisch); prüfen, ob der Server noch läuft.  
- **Server wartet endlos:** `min_available_clients` / `min_fit_clients` im Server-Notebook zu hoch  
  angesetzt — Wert an die tatsächliche Anzahl verbundener Clients anpassen.  
- **`Using start_server()`/`start_client()` is deprecated:** Erwartete Meldung, siehe Hinweis zur  
  Flower-API oben — kann ignoriert werden.

## Verwendete Frameworks

[Flower AI](https://flower.ai) · TensorFlow/Keras · [pyngrok](https://pyngrok.readthedocs.io) ·
matplotlib · numpy — ausdrücklich **ohne** TensorFlow Federated.

## Lizenz

Dieses Lehrprojekt kann frei für Lehrveranstaltungen verwendet, angepasst und weitergegeben
werden.
