import os

from flask import Flask, Response, jsonify, request, send_from_directory

from .config import ConfigValidationError
from .runtime import RuntimeService, mjpeg_stream


def create_app(config_path=None, runtime=None, auto_start=True):
    frontend_dist = _frontend_dist()
    static_folder = frontend_dist if os.path.isdir(frontend_dist) else None
    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    service = runtime or RuntimeService(config_path=config_path)
    app.config["runtime"] = service

    if auto_start and not app.config.get("TESTING"):
        service.start()

    @app.route("/")
    def index():
        if static_folder:
            return send_from_directory(static_folder, "index.html")
        return jsonify(
            {
                "name": "Jetson Nano YOLO Web",
                "message": "Build frontend assets or run the Vite dev server for the dashboard.",
            }
        )

    @app.route("/assets/<path:path>")
    def assets(path):
        if static_folder:
            return send_from_directory(os.path.join(static_folder, "assets"), path)
        return ("Not found", 404)

    @app.route("/stream.mjpg")
    def stream():
        return Response(mjpeg_stream(service), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def status():
        return jsonify(service.status())

    @app.route("/api/detections/latest")
    def latest_detections():
        return jsonify(service.latest_detections())

    @app.route("/api/config", methods=["GET"])
    def get_config():
        return jsonify(service.get_config())

    @app.route("/api/config", methods=["PUT"])
    def update_config():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON."}), 400
        try:
            return jsonify(service.update_config(payload))
        except ConfigValidationError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/snapshot", methods=["POST"])
    def snapshot():
        capture_dir = os.environ.get("JETSON_CAPTURE_DIR", os.path.abspath("captures"))
        return jsonify(service.save_snapshot(capture_dir)), 201

    @app.route("/healthz")
    def healthz():
        return jsonify({"ok": True, "health": service.status().get("health")})

    return app


def _frontend_dist():
    configured = os.environ.get("JETSON_FRONTEND_DIST")
    if configured:
        return os.path.abspath(configured)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "frontend", "dist"))


def main():
    config_path = os.environ.get("JETSON_CONFIG", "")
    app = create_app(config_path=config_path or None)
    host = os.environ.get("JETSON_HOST", "0.0.0.0")
    port = int(os.environ.get("JETSON_PORT", "8000"))
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
