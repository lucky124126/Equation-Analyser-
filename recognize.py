"""
Vercel Python serverless function: POST /api/recognize

Body: {"points": [[x, y], [x, y], ...]}  (math-space coordinates)
Returns the best-matching shape fit as JSON, same shape as the
Colab notebook's /recognize endpoint used.
"""

import json
from http.server import BaseHTTPRequestHandler

import numpy as np

# ---------------------------------------------------------------------
# Fitting engine (ported as-is from the notebook's fitting cell)
# ---------------------------------------------------------------------


def fit_line(points):
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, s, vt = np.linalg.svd(centered, full_matrices=False)

    direction = vt[0]
    normal = vt[1] if len(s) > 1 else np.array([-direction[1], direction[0]])

    residual = (s[1] ** 2 / len(pts)) if len(s) > 1 else 0.0
    rmse = float(np.sqrt(residual))

    a, b = normal
    c = a * centroid[0] + b * centroid[1]

    if abs(b) > 1e-9:
        slope = -a / b
        intercept = c / b
        eq = f"y = {slope:.3f}x + {intercept:.3f}"
    else:
        x0 = c / a
        eq = f"x = {x0:.3f}"

    return {"type": "line", "params": (a, b, c), "rmse": rmse, "equation": eq}


def fit_circle(points):
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(pts))])
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol

    cx, cy = -D / 2, -E / 2
    r = float(np.sqrt(max(cx ** 2 + cy ** 2 - F, 1e-9)))

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - r) ** 2)))

    cx_s = f"{cx:+.3f}".replace("+", "- ") if cx >= 0 else f"+ {abs(cx):.3f}"
    cy_s = f"{cy:+.3f}".replace("+", "- ") if cy >= 0 else f"+ {abs(cy):.3f}"
    if abs(cx) < 1e-3 and abs(cy) < 1e-3:
        eq = f"x^2 + y^2 = {r**2:.3f}   (r = {r:.3f})"
    else:
        eq = f"(x {cx_s})^2 + (y {cy_s})^2 = {r**2:.3f}   (center=({cx:.2f},{cy:.2f}), r={r:.3f})"

    return {"type": "circle", "params": (cx, cy, r), "rmse": rmse, "equation": eq}


def fit_parabola(points):
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]

    cy = np.polyfit(x, y, 2)
    rmse_y = float(np.sqrt(np.mean((y - np.polyval(cy, x)) ** 2)))

    cx = np.polyfit(y, x, 2)
    rmse_x = float(np.sqrt(np.mean((x - np.polyval(cx, y)) ** 2)))

    if rmse_y <= rmse_x:
        a, b, c = cy
        eq = f"y = {a:.3f}x^2 + {b:.3f}x + {c:.3f}"
        return {"type": "parabola", "orientation": "y(x)", "params": cy,
                "rmse": rmse_y, "equation": eq}
    else:
        a, b, c = cx
        eq = f"x = {a:.3f}y^2 + {b:.3f}y + {c:.3f}"
        return {"type": "parabola", "orientation": "x(y)", "params": cx,
                "rmse": rmse_x, "equation": eq}


def fit_ellipse(points):
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]

    M = np.column_stack([x ** 2, x * y, y ** 2, x, y, np.ones(len(pts))])
    scale = np.abs(M).max()
    M_n = M / scale
    _, _, vt = np.linalg.svd(M_n)
    A, B, C, D, E, F = vt[-1]

    disc = B ** 2 - 4 * A * C
    result = {"type": "ellipse" if disc < 0 else "conic",
              "params": (A, B, C, D, E, F), "rmse": None, "equation": None,
              "discriminant": disc}

    if disc >= 0:
        result["rmse"] = float("inf")
        result["equation"] = "not a valid ellipse"
        return result

    denom = disc
    x0 = (2 * C * D - B * E) / denom
    y0 = (2 * A * E - B * D) / denom

    Q = np.array([[A, B / 2], [B / 2, C]])
    const = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 + D * x0 + E * y0 + F
    eigvals, eigvecs = np.linalg.eigh(Q)
    axes = np.sqrt(np.abs(-const / eigvals))
    a_len, b_len = sorted(axes, reverse=True)

    theta = np.degrees(np.arctan2(eigvecs[1, np.argmin(eigvals)],
                                   eigvecs[0, np.argmin(eigvals)]))

    pred = M @ np.array([A, B, C, D, E, F])
    geo_rmse = float(np.sqrt(np.mean(pred ** 2)) / max(abs(A) + abs(C), 1e-9))

    result["rmse"] = geo_rmse
    result["center"] = (x0, y0)
    result["semi_axes"] = (a_len, b_len)
    result["angle_deg"] = theta
    if abs(x0) < 1e-2 and abs(y0) < 1e-2 and abs(theta) < 3:
        result["equation"] = f"x^2/{a_len**2:.3f} + y^2/{b_len**2:.3f} = 1"
    else:
        result["equation"] = (f"ellipse: center=({x0:.2f},{y0:.2f}), "
                               f"semi-axes=({a_len:.2f},{b_len:.2f}), "
                               f"rot={theta:.1f} deg")
    return result


def _shape_scale(points):
    pts = np.asarray(points, dtype=float)
    span = pts.max(axis=0) - pts.min(axis=0)
    return max(float(np.linalg.norm(span)), 1e-6)


def recognize_shape(points, min_points=5):
    if len(points) < min_points:
        return {"type": "unknown", "equation": "Not enough points", "confidence": 0.0}

    scale = _shape_scale(points)

    candidates = []
    for fitter in (fit_line, fit_circle, fit_parabola, fit_ellipse):
        try:
            candidates.append(fitter(points))
        except Exception:
            continue

    for c in candidates:
        c["normalized_error"] = c["rmse"] / scale if np.isfinite(c["rmse"]) else float("inf")

    priority = {"line": 0, "circle": 1, "parabola": 2, "ellipse": 3}
    candidates.sort(key=lambda c: c["normalized_error"])

    best = candidates[0]
    SIMPLICITY_MARGIN = 0.03
    for c in candidates:
        if (c["normalized_error"] <= best["normalized_error"] + SIMPLICITY_MARGIN
                and priority[c["type"] if c["type"] != "conic" else "ellipse"]
                    < priority[best["type"] if best["type"] != "conic" else "ellipse"]):
            best = c

    confidence = max(0.0, 1.0 - best["normalized_error"] / 0.25)
    best["confidence"] = round(min(confidence, 1.0), 2)
    return best


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (tuple, list)):
        return [to_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------
# Vercel entrypoint: a class named `handler` extending BaseHTTPRequestHandler
# ---------------------------------------------------------------------


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length else b'{}'
            data = json.loads(body or b'{}')
            points = data.get('points', [])

            result = recognize_shape(points)
            payload = json.dumps(to_jsonable(result)).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(err)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
