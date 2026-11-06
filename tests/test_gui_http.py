"""FT-10 (subida de datasets) y FT-13 (--host ::1) a través de HTTP real."""
import http.client
import json
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

TRAINER_DIR = Path(__file__).resolve().parents[1] / "trainer"
sys.path.insert(0, str(TRAINER_DIR))
import gui_server  # noqa: E402

VALID_LINE = json.dumps({"messages": [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "chao"}]})
DEEP_LINE = '{"messages":' + "[" * 100_000 + "]" * 100_000 + "}"


def request(host, port, method, path, raw=None, body=None):
    connection = http.client.HTTPConnection(host, port, timeout=10)
    name = f"[{host}]" if ":" in host else host
    headers = {"Host": f"{name}:{port}", "X-Fine-Tune-Token": gui_server.CSRF_TOKEN}
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
    if raw is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    try:
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data) if data else None
    finally:
        connection.close()


class DatasetUploadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.trainer = Path(temporary.name) / "trainer"
        self.datasets = self.trainer / "datasets"
        for name, value in (("TRAINER_DIR", self.trainer), ("DATASET_DIR", self.datasets)):
            patcher = mock.patch.object(gui_server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.server = gui_server.ThreadingHTTPServer(("127.0.0.1", 0), gui_server.Handler)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def upload(self, content, filename="datos.jsonl"):
        return request("127.0.0.1", self.port, "POST", "/api/datasets", body={"filename": filename, "content": content})

    def test_deeply_nested_dataset_returns_json_error_and_removes_temporary(self):
        status, body = self.upload(DEEP_LINE + "\n")

        self.assertEqual(status, 422)
        self.assertIn("No se pudo analizar el dataset", body["error"])
        self.assertEqual(list(self.datasets.iterdir()), [])

    def test_deeply_nested_request_body_returns_json_error(self):
        raw = ('{"filename": "datos.jsonl", "content": ' + "[" * 100_000 + "]" * 100_000 + "}").encode()

        status, body = request("127.0.0.1", self.port, "POST", "/api/datasets", raw=raw)

        self.assertEqual((status, body), (400, {"error": "JSON inválido."}))

    def test_dataset_nested_over_512_levels_is_rejected_on_any_stack(self):
        line = '{"messages":' + "[" * 512 + "]" * 512 + "}"

        status, body = self.upload(line + "\n")

        self.assertEqual(status, 422)
        self.assertEqual(body, {"error": "No se pudo analizar el dataset: el JSON está demasiado anidado."})
        self.assertEqual(list(self.datasets.iterdir()), [])

    def test_request_body_nested_over_512_levels_is_invalid_json(self):
        raw = ('{"filename": "datos.jsonl", "content": ' + "[" * 512 + "]" * 512 + "}").encode()

        status, body = request("127.0.0.1", self.port, "POST", "/api/datasets", raw=raw)

        self.assertEqual((status, body), (400, {"error": "JSON inválido."}))

    def test_request_body_nested_exactly_512_levels_is_still_parsed(self):
        raw = ('{"filename": "datos.jsonl", "content": ' + "[" * 511 + "]" * 511 + "}").encode()

        status, body = request("127.0.0.1", self.port, "POST", "/api/datasets", raw=raw)

        self.assertEqual((status, body), (400, {"error": "Contenido de dataset inválido."}))

    def test_brackets_inside_strings_do_not_count_as_nesting(self):
        brackets = "[{" * 600 + '\\"' + "}]" * 600
        line = json.dumps({"messages": [
            {"role": "user", "content": brackets},
            {"role": "assistant", "content": "chao"},
        ]})

        status, body = self.upload(line + "\n")

        self.assertEqual(status, 201)
        self.assertEqual(body["validation"]["valid"], True)

    def test_invalid_dataset_keeps_previous_message(self):
        status, body = self.upload('{"messages": []}\n')

        self.assertEqual(status, 422)
        self.assertEqual(body, {"error": "Línea 1: messages debe contener entre 2 y 64 elementos"})
        self.assertEqual(list(self.datasets.iterdir()), [])

    def test_valid_dataset_is_stored_without_temporaries(self):
        status, body = self.upload(VALID_LINE + "\n")

        self.assertEqual(status, 201)
        self.assertEqual(body, {
            "dataset": {"id": "datasets/datos.jsonl", "name": "datos.jsonl"},
            "validation": {"valid": True, "examples": 1, "messages": 2, "characters": 8},
        })
        self.assertEqual([path.name for path in self.datasets.iterdir()], ["datos.jsonl"])

    def test_write_failure_removes_temporary(self):
        with mock.patch.object(gui_server.os, "replace", side_effect=OSError("disco lleno")):
            status, body = self.upload(VALID_LINE + "\n")

        self.assertEqual(status, 500)
        self.assertIn("disco lleno", body["error"])
        self.assertEqual(list(self.datasets.iterdir()), [])


HARNESS = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import gui_server
    root = Path(sys.argv[2])
    gui_server.ROOT = root
    gui_server.RUNTIME_DIR = root / ".runtime"
    gui_server.ADAPTER_DIR = root / "adapters"
    gui_server.fetch_vllm_models = lambda: (False, [])
    sys.argv = ["gui_server.py", *sys.argv[3:]]
    gui_server.main()
    """
)


def free_port(family, host):
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def ipv6_loopback_available():
    if not socket.has_ipv6:
        return False
    try:
        free_port(socket.AF_INET6, "::1")
    except OSError:
        return False
    return True


class HostBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.harness = self.root / "harness.py"
        self.harness.write_text(HARNESS, encoding="utf-8")

    def launch(self, *args):
        process = subprocess.Popen(
            [sys.executable, "-u", str(self.harness), str(TRAINER_DIR), str(self.root), *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.addCleanup(lambda: (process.poll() is None and process.kill(), process.wait()))
        return process

    def wait_banner(self, process):
        lines = []
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            if line.startswith("Fine-tuning GUI:"):
                return lines
        process.wait()
        self.fail("\n".join(lines))

    @unittest.skipUnless(ipv6_loopback_available(), "requiere IPv6 en loopback")
    def test_ipv6_loopback_serves_requests(self):
        port = free_port(socket.AF_INET6, "::1")
        process = self.launch("--host", "::1", "--port", str(port))

        lines = self.wait_banner(process)

        self.assertIn(f"Fine-tuning GUI: http://[::1]:{port}", lines)
        self.assertEqual(request("::1", port, "GET", "/api/health"), (200, {"ok": True, "localOnly": True}))
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)

    def test_ipv4_loopback_keeps_serving(self):
        port = free_port(socket.AF_INET, "127.0.0.1")
        process = self.launch("--port", str(port))

        lines = self.wait_banner(process)

        self.assertIn(f"Fine-tuning GUI: http://127.0.0.1:{port}", lines)
        self.assertIn("Acceso restringido a la máquina local.", "\n".join(lines + [process.stdout.readline()]))
        self.assertEqual(request("127.0.0.1", port, "GET", "/api/health"), (200, {"ok": True, "localOnly": True}))
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)

    def test_non_loopback_host_is_still_rejected(self):
        process = self.launch("--host", "0.0.0.0", "--port", "0")

        output, _ = process.communicate(timeout=10)

        self.assertEqual(process.returncode, 1)
        self.assertIn("Por seguridad, la GUI solo puede enlazarse a loopback.", output)


if __name__ == "__main__":
    unittest.main()
