# Edge-AI

# ROS 2 YOLO Object Detection (Proof of Concept)

Dieses Repository enthält den Proof of Concept eines kamerabasierten Objekterkennungssystems für ein autonomes Modellfahrzeug. Das Projekt wurde im Rahmen eines Digitalisierungsprojekts umgesetzt und basiert auf **ROS 2 (Humble)**. Ziel ist die lokale, latenzarme Wahrnehmung von Verkehrsobjekten auf einem mobilen Versuchsträger.

## Projektüberblick

Das System nutzt eine GPU-beschleunigte Edge-Plattform zur KI-Inferenz und ist modular als ROS‑2‑Workspace aufgebaut. Die Objekterkennung erfolgt mithilfe eines YOLO-Modells, wobei erkannte Objekte visualisiert, dedupliziert und persistent gespeichert werden. Die Architektur ist so ausgelegt, dass Wahrnehmung, Entscheidungslogik und Aktorik klar voneinander getrennt sind.

## Hardware

* **NVIDIA Jetson Orin Nano** – Edge-Rechner für KI-Inferenz und High-Level-Logik
* **Intel RealSense D455** – RGB‑D‑Kamera zur Erfassung von Farb- und Tiefeninformationen
* **ESP32** – Mikrocontroller für Aktorsteuerung und zeitkritische Aufgaben
* **Modellfahrzeug (1:10)** mit BLDC‑Motor und Lenkservo

## Software-Stack

* Ubuntu 22.04.5 LTS
* ROS 2 Humble
* NVIDIA JetPack SDK 6.2
* Python 3.10.12
* Ultralytics YOLO (v12)
* OpenCV 4.12.0

## Funktionalitäten

* Initialisierung und Ausführung eines YOLO‑Objektdetektors
* Kamerabildaufnahme und ROI‑basierte Objekterkennung
* Live‑Visualisierung mit Bounding Boxes, Klassen, Konfidenzen und FPS
* Umschalten zwischen Full‑Frame‑ und ROI‑Detektion zur Laufzeit
* Deduplizierung erkannter Objekte auf Basis ihrer Bildposition
* Persistente Speicherung der Detektionen in einer CSV‑Datei

## Deduplizierung und Logging

Um Mehrfachzählungen zu vermeiden, wird für jede Detektion der Mittelpunkt der Bounding Box berechnet. Detektionen derselben Klasse werden nur dann gespeichert, wenn ihr Abstand zu bereits bekannten Objekten einen definierten Schwellwert überschreitet. Neue Objekte werden mit Zeitstempel, Klasse, Konfidenz und Bounding‑Box‑Koordinaten in einer CSV‑Datei abgelegt. Beim Neustart des Programms wird diese Datei erneut eingelesen.

## Repository-Struktur

* `build/` – Build-Artefakte, erzeugt durch `colcon build`
* `install/` – Installationsverzeichnis des ROS‑2‑Workspaces (Setup-Skripte, installierte Packages)
* `log/` – Build- und Laufzeit-Logs von ROS 2 und colcon
* `runs/` – Laufzeitdaten des YOLO‑Frameworks (z. B. Inferenz‑Outputs, temporäre Ergebnisse)
* `src/` – Quellcode der ROS‑2‑Packages (Python‑Nodes, Konfigurationen, Launch-Dateien)

## Ausführung

1. ROS‑2‑Workspace bauen:

   ```bash
   colcon build
   source install/setup.bash
   ```
2. Node starten:

   ```bash
   ros2 run ros2_yolo_rs image_publisher
   rqt_image_view
   ```

## Zielsetzung

Das Projekt dient als praxisnahe Demonstration einer vollständig lokalen Wahrnehmungskette für mobile Robotik. Es bildet die Grundlage für weiterführende Arbeiten in den Bereichen Fahrentscheidungslogik, Sensorfusion und autonome Navigation.

## Lizenz

Dieses Projekt dient Forschungs- und Lehrzwecken. Die Lizenzbedingungen der verwendeten Bibliotheken (ROS 2, Ultralytics YOLO, OpenCV) sind zu beachten.
