#!/usr/bin/env python3
"""
Script de referencia para la Fase 1 (fix ASCII)
"""
from __future__ import annotations
import cv2
import time
import argparse
from pathlib import Path
import sys

try:
    import mediapipe as mp
except ImportError as e:
    print("ERROR: No se encontro MediaPipe. Instala con: pip install mediapipe", file=sys.stderr)
    raise

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def _backend_display_name(backend: int | None) -> str:
    if backend is None:
        return "default"
    if hasattr(cv2, "CAP_AVFOUNDATION") and backend == cv2.CAP_AVFOUNDATION:
        return "AVFoundation"
    if hasattr(cv2, "CAP_DSHOW") and backend == cv2.CAP_DSHOW:
        return "DirectShow"
    if hasattr(cv2, "CAP_V4L2") and backend == cv2.CAP_V4L2:
        return "V4L2"
    return str(backend)

def _open_camera_device(camera_index: int, width: int | None = None, height: int | None = None, warmup_reads: int = 20, retry_delay: float = 0.05):
    candidates = []
    if camera_index is not None:
        candidates.append(camera_index)
    for idx in range(5):
        if idx not in candidates:
            candidates.append(idx)
    backend_candidates: list[int | None] = [None]
    for attr in ("CAP_AVFOUNDATION", "CAP_DSHOW", "CAP_V4L2"):
        backend_value = getattr(cv2, attr, None)
        if backend_value is not None and backend_value not in backend_candidates:
            backend_candidates.append(backend_value)
    for idx in candidates:
        for backend in backend_candidates:
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ready = False
            for _ in range(warmup_reads):
                ret, frame = cap.read()
                if ret and frame is not None:
                    ready = True
                    break
                time.sleep(retry_delay)
            if ready:
                return cap, idx, backend
            cap.release()
    return None, None, None

def _make_writer(path: Path, width: int, height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps if fps > 0 else 30.0, (width, height))

def record_camera(duration_sec: int = 20, output: Path = Path("raw20.mp4"), camera_index: int = 0, width: int | None = None, height: int | None = None):
    cap, used_index, backend = _open_camera_device(camera_index, width, height)
    if cap is None:
        print("ERROR: No se pudo abrir la camara. Verifica permisos en macOS (Privacidad > Camara) y que ninguna otra aplicacion la este utilizando.")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(output)
    writer = _make_writer(output, w, h, fps)
    print(f"Grabando {duration_sec}s a {fps:.1f} FPS aprox. -> {output} (camara {used_index}, backend {_backend_display_name(backend)})")
    start = time.monotonic()
    try:
        empty_reads = 0
        max_empty_reads = 50
        while True:
            ret, frame = cap.read()
            if not ret:
                empty_reads += 1
                if empty_reads >= max_empty_reads:
                    print("Fin o error de captura: la camara no entrega frames. Revisa permisos o intenta con otro indice (--camera-index).")
                    break
                time.sleep(0.05)
                continue
            empty_reads = 0
            writer.write(frame)
            cv2.imshow("Grabando (q/Esc para salir)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Grabacion detenida por el usuario (tecla).")
                break
            if time.monotonic() - start >= duration_sec:
                break
    except KeyboardInterrupt:
        print("\nGrabacion interrumpida con Ctrl+C; cerrando archivos...")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print("Listo.")
    return 0

def process_image(input_path: Path, output_path: Path | None = None, draw_landmarks: bool = True):
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        print(f"ERROR: no se pudo leer {input_path}")
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
    if results.pose_landmarks and draw_landmarks:
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    if output_path:
        cv2.imwrite(str(output_path), image_bgr)
        print(f"Salida guardada en {output_path}")
    cv2.imshow("Pose en imagen (q para cerrar)", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def process_video(input_path: Path, output_path: Path = Path("processed.mp4"), show: bool = True, mirror: bool = True):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir {input_path}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _make_writer(Path(output_path), w, h, fps)
    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            prev_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                now = time.time()
                fps_now = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                writer.write(frame)
                if show:
                    cv2.imshow("Procesando video (q para salir)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        print("\nProcesado interrumpido con Ctrl+C; cerrando archivos...")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Video procesado guardado en {output_path}")
    return 0

def webcam_stream(camera_index: int = 0, mirror: bool = True, save_output: bool = False, output_path: Path = Path("webcam_pose.mp4")):
    cap, used_index, backend = _open_camera_device(camera_index, None, None)
    if cap is None:
        print("ERROR: No se pudo abrir la camara. Verifica permisos del sistema o prueba con --camera-index.")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _make_writer(Path(output_path), w, h, fps) if save_output else None
    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            prev_time = time.time()
            print("Transmision en tiempo real iniciada. Presiona 'q' para salir.")
            print(f"Camara en uso: indice {used_index} (backend {_backend_display_name(backend)}), resolucion {w}x{h} @ {fps:.1f} FPS aprox.")
            if save_output:
                print(f"Guardando transmision en {output_path}")
            empty_reads = 0
            max_empty_reads = 50
            while True:
                ret, frame = cap.read()
                if not ret:
                    empty_reads += 1
                    if empty_reads >= max_empty_reads:
                        print("Fin de la transmision o error de captura: la camara no esta entregando frames. Verifica permisos y que ninguna otra app la utilice.")
                        break
                    time.sleep(0.05)
                    continue
                empty_reads = 0
                if mirror:
                    frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                now = time.time()
                fps_now = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if writer is not None:
                    writer.write(frame)
                cv2.imshow("Pose en tiempo real (q para salir)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nTransmision interrumpida con Ctrl+C; cerrando archivos...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Transmision finalizada.")
    return 0

def build_arg_parser():
    p = argparse.ArgumentParser(description="Fase 1: captura, poses y transmision en tiempo real (OpenCV + MediaPipe)")
    sub = p.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("record", help="Graba N segundos desde la camara")
    sp.add_argument("--duration", type=int, default=20, help="Duracion en segundos (por defecto 20)")
    sp.add_argument("--output", type=Path, default=Path("raw20.mp4"), help="Ruta de salida del video")
    sp.add_argument("--camera-index", type=int, default=0, help="Indice de la camara (por defecto 0)")
    sp.add_argument("--width", type=int, default=None, help="Ancho deseado de la captura")
    sp.add_argument("--height", type=int, default=None, help="Alto deseado de la captura")
    sp = sub.add_parser("image", help="Detecta y dibuja pose en una imagen")
    sp.add_argument("--input", type=Path, required=True, help="Ruta de imagen")
    sp.add_argument("--output", type=Path, default=None, help="Ruta de salida de la imagen con overlay")
    sp = sub.add_parser("process-video", help="Procesa un video y dibuja el esqueleto en cada frame")
    sp.add_argument("--input", type=Path, required=True, help="Ruta de video de entrada")
    sp.add_argument("--output", type=Path, default=Path("processed.mp4"), help="Ruta de salida del video procesado")
    sp.add_argument("--no-show", action="store_true", help="No mostrar ventana durante el procesado")
    sp.add_argument("--no-mirror", action="store_true", help="No espejar frames (por defecto se espeja)")
    sp = sub.add_parser("webcam", help="Transmision en tiempo real con esqueleto")
    sp.add_argument("--camera-index", type=int, default=0, help="Indice de la camara (por defecto 0)")
    sp.add_argument("--no-mirror", action="store_true", help="No espejar la vista de camara")
    sp.add_argument("--save-output", action="store_true", help="Guardar la transmision en un mp4")
    sp.add_argument("--output", type=Path, default=Path("webcam_pose.mp4"), help="Ruta del mp4 de salida si se guarda")
    return p

def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "record":
        return record_camera(duration_sec=args.duration, output=args.output, camera_index=args.camera_index, width=args.width, height=args.height)
    if args.command == "image":
        return process_image(input_path=args.input, output_path=args.output)
    if args.command == "process-video":
        return process_video(input_path=args.input, output_path=args.output, show=not args.no_show, mirror=not args.no_mirror)
    if args.command == "webcam":
        return webcam_stream(camera_index=args.camera_index, mirror=not args.no_mirror, save_output=args.save_output, output_path=args.output)
    parser.print_help()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
