"""
manim_srfm.py -- Animated Minkowski spacetime classification video.

Generates a 60-second educational animation explaining SRFM physics:
1. Spacetime metric introduction
2. Price return --> beta calculation
3. TIMELIKE vs SPACELIKE classification on light cone
4. BH mass accrual animation
5. BH formation and convergence event

Uses Manim Community Edition (pip install manim).
Falls back to SVG static frames if manim not available.

Usage:
    python tools/manim_srfm.py             # generate SVG storyboard
    manim -pql tools/manim_srfm.py SRFM   # render full video (low quality)
    manim -pqh tools/manim_srfm.py SRFM   # high quality
"""

import argparse
import math
import os
import sys

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
FRAMES_DIR    = os.path.join(RESULTS_DIR, "manim_frames")


# -- SVG helpers ----------------------------------------------------------------

def svg_header(w: int = 800, h: int = 600) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n'
            f'<rect width="{w}" height="{h}" fill="white"/>\n')

def svg_footer() -> str:
    return "</svg>\n"

def svg_text(x, y, text, size=18, color="black", bold=False, anchor="middle"):
    weight = "bold" if bold else "normal"
    return (f'<text x="{x}" y="{y}" font-family="monospace" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{text}</text>\n')

def svg_line(x1, y1, x2, y2, color="black", width=2, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash_attr}/>\n'

def svg_circle(cx, cy, r, fill="blue", stroke="black", sw=1):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'

def svg_rect(x, y, w, h, fill="#4a90d9", stroke="none", rx=4):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" rx="{rx}"/>\n'

def svg_poly(points, fill="none", stroke="black", sw=2):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'

def svg_arrow(x1, y1, x2, y2, color="black", width=2):
    # Arrow line + simple arrowhead
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy) or 1
    ux, uy = dx/length, dy/length
    # Arrowhead
    hw = 8
    hlen = 12
    bx, by = x2 - hlen*ux, y2 - hlen*uy
    p1 = (bx - hw*uy, by + hw*ux)
    p2 = (bx + hw*uy, by - hw*ux)
    head = f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>\n'
    line = svg_line(x1, y1, x2, y2, color, width)
    return line + head


# -- Frame 1: Minkowski Metric -------------------------------------------------

def frame_01_metric() -> str:
    s = svg_header()
    s += svg_text(400, 50, "Frame 1 -- Minkowski Spacetime Metric", 22, bold=True)

    # Metric equation
    s += svg_text(400, 110, "ds² = -c²dt² + dx²", 32, "#1a1a8c", bold=True)
    s += svg_text(400, 150, "In SRFM: c = cf (classification frequency)", 16, "#555")

    # Light cone diagram
    cx, cy = 400, 360
    cone_h = 180

    # Past cone (below)
    s += f'<polygon points="{cx},{cy} {cx-cone_h},{cy+cone_h} {cx+cone_h},{cy+cone_h}" '
    s += 'fill="#ffe0e0" stroke="#e74c3c" stroke-width="2"/>\n'
    # Future cone (above)
    s += f'<polygon points="{cx},{cy} {cx-cone_h},{cy-cone_h} {cx+cone_h},{cy-cone_h}" '
    s += 'fill="#e0f0ff" stroke="#3498db" stroke-width="2"/>\n'

    # Spacelike region (left and right)
    s += f'<polygon points="{cx},{cy} {cx-cone_h},{cy-cone_h} {cx-cone_h},{cy+cone_h}" '
    s += 'fill="#f0ffe0" stroke="none"/>\n'
    s += f'<polygon points="{cx},{cy} {cx+cone_h},{cy-cone_h} {cx+cone_h},{cy+cone_h}" '
    s += 'fill="#f0ffe0" stroke="none"/>\n'

    # Axes
    s += svg_arrow(cx, cy+200, cx, cy-210, "#333", 2)
    s += svg_arrow(cx-220, cy, cx+220, cy, "#333", 2)
    s += svg_text(cx, cy-220, "time (t)", 14, "#333")
    s += svg_text(cx+230, cy+5, "space (x)", 14, "#333")

    # Labels
    s += svg_text(cx, cy-90, "FUTURE CONE", 13, "#1a6ebf", bold=True)
    s += svg_text(cx, cy+110, "PAST CONE",  13, "#c0392b", bold=True)
    s += svg_text(cx-140, cy+15, "SPACELIKE", 12, "#27ae60", bold=True)
    s += svg_text(cx+140, cy+15, "SPACELIKE", 12, "#27ae60", bold=True)

    # Origin dot
    s += svg_circle(cx, cy, 5, "black")

    s += svg_text(400, 570, "Light cone separates TIMELIKE events (inside) from SPACELIKE (outside)", 13, "#777")
    s += svg_footer()
    return s


# -- Frame 2: Beta Calculation -------------------------------------------------

def frame_02_beta() -> str:
    s = svg_header()
    s += svg_text(400, 50, "Frame 2 -- Price Return --> Beta Calculation", 22, bold=True)

    s += svg_text(400, 110, "β = |Δp/p| / cf", 36, "#1a1a8c", bold=True)
    s += svg_text(400, 150, "cf = classification frequency (e.g. 0.001)", 16, "#555")

    # Number line showing beta values
    lx, ly = 80, 300
    s += svg_arrow(lx, ly, 720, ly, "#333", 2)

    # Tick marks
    for i, (val, label) in enumerate([(0, "0"), (0.5, "0.5"), (1.0, "1.0 (=cf)"),
                                       (1.5, "1.5"), (2.0, "2.0"), (3.0, "3.0")]):
        x = lx + val * 200
        if x <= 720:
            s += svg_line(x, ly-8, x, ly+8, "#333", 2)
            s += svg_text(x, ly+28, label, 12, "#333")

    # Timelike zone (beta < 1)
    s += svg_rect(lx, ly-40, 200, 30, "#b3d9ff", "none")
    s += svg_text(lx+100, ly-18, "TIMELIKE  (β < 1)", 13, "#1a6ebf", bold=True)

    # Spacelike zone (beta > 1)
    s += svg_rect(lx+200, ly-40, 500, 30, "#ffd5b3", "none")
    s += svg_text(lx+450, ly-18, "SPACELIKE  (β > 1)", 13, "#c0392b", bold=True)

    # Dividing line at beta=1
    s += svg_line(lx+200, ly-50, lx+200, ly+50, "#e74c3c", 3, "6,3")
    s += svg_text(lx+200, ly+65, "Light cone boundary", 12, "#e74c3c")

    # Example calculation
    s += svg_text(400, 420, "Example:", 16, "#333", bold=True)
    s += svg_text(400, 450, "close_prev = 4500.00   close = 4545.00", 15, "#444")
    s += svg_text(400, 478, "Δp/p = (4545-4500)/4500 = 0.010", 15, "#444")
    s += svg_text(400, 506, "β = 0.010 / 0.001 = 10.0  --> SPACELIKE (fast move!)", 15, "#c0392b", bold=True)

    s += svg_text(400, 570, "Only TIMELIKE bars (slow/steady) accrue BH mass", 14, "#555")
    s += svg_footer()
    return s


# -- Frame 3: TIMELIKE / SPACELIKE classification ------------------------------

def frame_03_classification() -> str:
    s = svg_header()
    s += svg_text(400, 50, "Frame 3 -- TIMELIKE vs SPACELIKE Classification", 22, bold=True)

    cx, cy = 400, 310
    cone_h = 200

    # Draw light cone
    s += f'<polygon points="{cx},{cy} {cx-cone_h},{cy-cone_h} {cx+cone_h},{cy-cone_h}" fill="#d0e8ff" stroke="#3498db" stroke-width="2"/>\n'
    s += f'<polygon points="{cx},{cy} {cx-cone_h},{cy+cone_h} {cx+cone_h},{cy+cone_h}" fill="#d0e8ff" stroke="#3498db" stroke-width="2"/>\n'

    # TIMELIKE events (inside cone, above)
    timelike_pts = [(cx-50, cy-80), (cx+30, cy-120), (cx-20, cy-160)]
    for px, py in timelike_pts:
        s += svg_circle(px, py, 10, "#27ae60", "white", 2)
        s += svg_text(px+18, py+5, "TIMELIKE", 11, "#27ae60", bold=True, anchor="start")

    # SPACELIKE events (outside cone)
    spacelike_pts = [(cx-180, cy-60), (cx+190, cy+50), (cx-160, cy+100)]
    for px, py in spacelike_pts:
        s += svg_circle(px, py, 10, "#e74c3c", "white", 2)
        s += svg_text(px+18, py+5, "SPACELIKE", 11, "#e74c3c", bold=True, anchor="start")

    # Origin
    s += svg_circle(cx, cy, 6, "black")
    s += svg_text(cx+12, cy+5, "Origin (current bar)", 12, "#333", anchor="start")

    # Light cone lines
    s += svg_line(cx, cy, cx-cone_h, cy-cone_h, "#3498db", 2)
    s += svg_line(cx, cy, cx+cone_h, cy-cone_h, "#3498db", 2)

    # Labels
    s += svg_text(cx, cy-cone_h-20, "Light cone boundary: β = 1.0", 14, "#3498db")
    s += svg_text(cx, cy+cone_h+25, "Spacelike region: fast price moves, no BH mass", 14, "#e74c3c")

    # Decision box
    s += svg_rect(50, 520, 700, 50, "#f8f8f8", "#ccc")
    s += svg_text(400, 549, "IF β < 1.0 --> TIMELIKE --> accrue BH mass     IF β ≥ 1.0 --> SPACELIKE --> decay/hold", 14, "#333")

    s += svg_footer()
    return s


# -- Frame 4: BH Mass Accrual --------------------------------------------------

def frame_04_mass() -> str:
    s = svg_header()
    s += svg_text(400, 45, "Frame 4 -- Black Hole Mass Accrual", 22, bold=True)

    # Simulate some BH mass values
    mass_vals = [0.0, 0.3, 0.6, 0.55, 0.85, 1.1, 1.0, 1.3, 1.5, 1.7, 1.6, 1.8, 2.0, 1.9, 1.5]
    is_timelike = [True, True, True, False, True, True, False, True, True, True, False, True, True, False, False]
    threshold = 1.5

    bw = 36
    gap = 4
    n = len(mass_vals)
    chart_left = 80
    chart_bottom = 450
    chart_height = 280
    max_mass = 2.2

    # Axes
    s += svg_arrow(chart_left-5, chart_bottom+10, chart_left-5, chart_bottom-chart_height-20, "#333", 2)
    s += svg_arrow(chart_left-5, chart_bottom+10, chart_left + n*(bw+gap)+20, chart_bottom+10, "#333", 2)
    s += svg_text(chart_left-20, chart_bottom-chart_height-30, "BH Mass", 14, "#333", anchor="start")

    # Threshold line
    threshold_y = chart_bottom - (threshold / max_mass) * chart_height
    s += svg_line(chart_left-5, threshold_y, chart_left + n*(bw+gap)+15, threshold_y, "#e74c3c", 2, "8,4")
    s += svg_text(chart_left + n*(bw+gap)+20, threshold_y+5, f"Threshold={threshold}", 12, "#e74c3c", anchor="start")

    # Y-axis ticks
    for tick in [0.0, 0.5, 1.0, 1.5, 2.0]:
        ty = chart_bottom - (tick / max_mass) * chart_height
        s += svg_line(chart_left-10, ty, chart_left-5, ty, "#333", 1)
        s += svg_text(chart_left-15, ty+5, f"{tick:.1f}", 11, "#333", anchor="end")

    # Bars
    for i, (mv, tl) in enumerate(zip(mass_vals, is_timelike)):
        bx = chart_left + i * (bw + gap)
        bh_bar = (mv / max_mass) * chart_height
        by = chart_bottom - bh_bar
        color = "#27ae60" if tl else "#aaa"
        s += svg_rect(bx, by, bw, bh_bar, color, "none", 2)
        label = "TL" if tl else "SL"
        s += svg_text(bx + bw//2, chart_bottom + 22, label, 10, "#666")

    # BH formation annotation
    form_idx = 8  # where mass first crosses 1.5
    form_x = chart_left + form_idx * (bw + gap) + bw//2
    s += svg_arrow(form_x, chart_bottom - chart_height - 10, form_x, threshold_y - 5, "#8e44ad", 2)
    s += svg_text(form_x, chart_bottom - chart_height - 18, "BH Forms!", 13, "#8e44ad", bold=True)

    s += svg_text(400, 545, "Green bars = TIMELIKE (β<1), Gray = SPACELIKE (β≥1)", 13, "#555")
    s += svg_text(400, 570, "BH activates when cumulative TIMELIKE mass ≥ 1.5", 13, "#8e44ad")
    s += svg_footer()
    return s


# -- Frame 5: BH Formation -----------------------------------------------------

def frame_05_formation() -> str:
    s = svg_header()
    s += svg_text(400, 45, "Frame 5 -- Black Hole Formation Event", 22, bold=True)

    # Mass gauge
    cx, cy_gauge = 200, 280
    radius = 130
    s += f'<circle cx="{cx}" cy="{cy_gauge}" r="{radius}" fill="#f0f0f0" stroke="#ccc" stroke-width="3"/>\n'

    # Fill arc proportional to mass (simulate mass=1.7, threshold=1.5)
    mass = 1.7
    threshold = 1.5
    fill_pct = min(1.0, mass / 2.0)
    # Draw filled sector
    angle = fill_pct * 2 * math.pi
    end_x = cx + radius * math.sin(angle)
    end_y = cy_gauge - radius * math.cos(angle)
    large_arc = 1 if angle > math.pi else 0
    s += (f'<path d="M {cx},{cy_gauge} L {cx},{cy_gauge-radius} '
          f'A {radius},{radius} 0 {large_arc},1 {end_x:.1f},{end_y:.1f} Z" '
          f'fill="#8e44ad" opacity="0.7"/>\n')

    # Threshold marker
    thresh_pct = threshold / 2.0
    thresh_angle = thresh_pct * 2 * math.pi
    tx = cx + radius * math.sin(thresh_angle)
    ty = cy_gauge - radius * math.cos(thresh_angle)
    s += svg_line(cx, cy_gauge, tx, ty, "#e74c3c", 3)
    s += svg_text(cx+5, cy_gauge-10, f"mass={mass:.1f}", 18, "white", bold=True)
    s += svg_text(cx, cy_gauge + radius + 25, "BH MASS GAUGE", 14, "#333")
    s += svg_text(cx, cy_gauge + radius + 45, f"Threshold = {threshold}", 13, "#e74c3c")

    # BH icon
    bh_x, bh_y = 570, 260
    # Event horizon
    s += f'<circle cx="{bh_x}" cy="{bh_y}" r="80" fill="black"/>\n'
    # Accretion disk glow
    for r_glow in [95, 85]:
        opacity = 0.3 if r_glow == 95 else 0.5
        s += (f'<circle cx="{bh_x}" cy="{bh_y}" r="{r_glow}" fill="none" '
              f'stroke="#f39c12" stroke-width="6" opacity="{opacity}"/>\n')
    s += svg_text(bh_x, bh_y + 5, "BH", 24, "white", bold=True)
    s += svg_text(bh_x, bh_y + 115, "BLACK HOLE ACTIVE", 14, "#8e44ad", bold=True)
    s += svg_text(bh_x, bh_y + 135, "Position sizing BOOSTED", 13, "#27ae60")

    # Arrow from gauge to BH
    s += svg_arrow(cx+radius+10, cy_gauge, bh_x-95, bh_y, "#27ae60", 3)
    s += svg_text(390, cy_gauge-20, "ACTIVATES", 14, "#27ae60", bold=True)

    # Conditions box
    s += svg_rect(50, 520, 700, 55, "#f0fff0", "#27ae60", 6)
    s += svg_text(400, 542, "BH activation conditions:", 14, "#333", bold=True)
    s += svg_text(400, 566, "bh_mass ≥ 1.5  AND  TIMELIKE streak ≥ tl_req  -->  enter position with conviction sizing", 13, "#333")

    s += svg_footer()
    return s


# -- Frame 6: Convergence Event ------------------------------------------------

def frame_06_convergence() -> str:
    s = svg_header()
    s += svg_text(400, 40, "Frame 6 -- Convergence Event (Multi-Instrument BH)", 22, bold=True)

    instruments = [
        {"name": "ES",  "x": 160, "color": "#2980b9", "mass": 1.8, "active": True},
        {"name": "NQ",  "x": 400, "color": "#27ae60", "mass": 2.1, "active": True},
        {"name": "YM",  "x": 640, "color": "#8e44ad", "mass": 1.6, "active": True},
    ]

    cy = 260
    r  = 70

    for inst in instruments:
        # BH circle
        s += f'<circle cx="{inst["x"]}" cy="{cy}" r="{r}" fill="black"/>\n'
        # Glow
        s += (f'<circle cx="{inst["x"]}" cy="{cy}" r="{r+15}" fill="none" '
              f'stroke="{inst["color"]}" stroke-width="8" opacity="0.6"/>\n')
        s += svg_text(inst["x"], cy+5, inst["name"], 22, "white", bold=True)
        s += svg_text(inst["x"], cy+r+20, f"mass={inst['mass']:.1f}", 14, inst["color"])
        s += svg_text(inst["x"], cy+r+38, "BH ACTIVE", 13, "#27ae60", bold=True)

    # Convergence arrows pointing toward center
    mid_y = cy + r + 80
    s += svg_arrow(instruments[0]["x"]+r+5, cy, 330, mid_y, "#f39c12", 3)
    s += svg_arrow(instruments[1]["x"],   cy+r+5, instruments[1]["x"], mid_y, "#f39c12", 3)
    s += svg_arrow(instruments[2]["x"]-r-5, cy, 470, mid_y, "#f39c12", 3)

    # Convergence star burst
    conv_x, conv_y = 400, mid_y + 60
    for angle_deg in range(0, 360, 30):
        angle = math.radians(angle_deg)
        x2 = conv_x + 50 * math.cos(angle)
        y2 = conv_y + 50 * math.sin(angle)
        s += svg_line(conv_x, conv_y, x2, y2, "#f39c12", 3)
    s += svg_circle(conv_x, conv_y, 20, "#f39c12", "white", 2)
    s += svg_text(conv_x, conv_y+6, "★", 20, "white")

    s += svg_text(conv_x, conv_y+75, "CONVERGENCE EVENT", 20, "#e67e22", bold=True)
    s += svg_text(conv_x, conv_y+98, "All 3 instruments BH-active simultaneously", 15, "#555")

    # Stats box
    s += svg_rect(50, 505, 700, 70, "#fff8e1", "#f39c12", 6)
    s += svg_text(400, 525, "LARSA v1 Historical Stats:", 14, "#333", bold=True)
    s += svg_text(400, 548, "47 convergence wells  |  Win Rate: 74.5%  |  Half-Kelly: 19.1%", 14, "#c0392b")
    s += svg_text(400, 570, "Solo wells: 216  |  Win Rate: 50.0%  |  Half-Kelly: 0.6%", 14, "#777")

    s += svg_footer()
    return s


# -- Manim Scene (if manim is available) ---------------------------------------

def try_import_manim():
    try:
        import manim
        return True
    except ImportError:
        return False


MANIM_SCENE_CODE = '''
try:
    from manim import *

    class SRFM(Scene):
        def construct(self):
            # -- Title ---------------------------------------------------------
            title = Text("SRFM: Spacetime-Relativistic\\nFinancial Mechanics",
                         font_size=40, color=BLUE)
            subtitle = Text("Minkowski metric applied to price returns",
                            font_size=24, color=GRAY).next_to(title, DOWN)
            self.play(Write(title), run_time=2)
            self.play(FadeIn(subtitle))
            self.wait(1)
            self.play(FadeOut(title), FadeOut(subtitle))

            # -- Metric equation -----------------------------------------------
            eq = MathTex(r"ds^2 = -c_f^2 dt^2 + dx^2", font_size=60)
            label = Text("Minkowski metric (cf = classification frequency)",
                         font_size=20, color=GRAY).next_to(eq, DOWN)
            self.play(Write(eq), run_time=2)
            self.play(FadeIn(label))
            self.wait(2)
            self.play(FadeOut(eq), FadeOut(label))

            # -- Light cone ----------------------------------------------------
            axes = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                        x_length=6, y_length=6)
            cone_future = Polygon(
                ORIGIN, axes.c2p(-2, 2), axes.c2p(2, 2),
                fill_color=BLUE, fill_opacity=0.3, stroke_color=BLUE
            )
            cone_past = Polygon(
                ORIGIN, axes.c2p(-2, -2), axes.c2p(2, -2),
                fill_color=RED, fill_opacity=0.3, stroke_color=RED
            )
            tl_label = Text("TIMELIKE", color=GREEN, font_size=22).move_to(axes.c2p(0, 1))
            sl_label = Text("SPACELIKE", color=RED, font_size=22).move_to(axes.c2p(2.2, 0))

            self.play(Create(axes), run_time=1)
            self.play(FadeIn(cone_future), FadeIn(cone_past), run_time=2)
            self.play(Write(tl_label), Write(sl_label))
            self.wait(2)
            self.play(FadeOut(axes), FadeOut(cone_future), FadeOut(cone_past),
                      FadeOut(tl_label), FadeOut(sl_label))

            # -- Beta calculation ----------------------------------------------
            beta_eq = MathTex(r"\\beta = \\frac{|\\Delta p / p|}{c_f}", font_size=60, color=YELLOW)
            self.play(Write(beta_eq), run_time=2)
            self.wait(1)
            rule = Text("β < 1 --> TIMELIKE --> accrue BH mass",
                        font_size=24, color=GREEN).next_to(beta_eq, DOWN)
            rule2 = Text("β ≥ 1 --> SPACELIKE --> decay mass",
                         font_size=24, color=RED).next_to(rule, DOWN)
            self.play(FadeIn(rule), FadeIn(rule2))
            self.wait(3)
            self.play(*[FadeOut(m) for m in [beta_eq, rule, rule2]])

            # -- BH mass accrual -----------------------------------------------
            mass_label = Text("Black Hole Mass Accrual", font_size=36, color=PURPLE)
            self.play(Write(mass_label))
            self.wait(1)
            self.play(FadeOut(mass_label))

            threshold_line = DashedLine(
                start=LEFT*3, end=RIGHT*3, color=RED
            ).shift(UP*0.5)
            thresh_label = Text("threshold = 1.5", font_size=18, color=RED)
            thresh_label.next_to(threshold_line, RIGHT)
            self.play(Create(threshold_line), Write(thresh_label))

            bars = VGroup()
            masses = [0.2, 0.5, 0.8, 0.7, 1.1, 1.4, 1.6, 1.9]
            for i, m in enumerate(masses):
                bar = Rectangle(width=0.4, height=m, fill_color=GREEN if m < 1.5 else PURPLE,
                                 fill_opacity=0.8, stroke_width=1)
                bar.align_to(ORIGIN, DOWN).shift(LEFT*3.5 + RIGHT*(i*0.6))
                bars.add(bar)

            for bar in bars:
                self.play(GrowFromEdge(bar, DOWN), run_time=0.3)
            self.wait(1)

            bh_form = Text("BH FORMED!", font_size=30, color=PURPLE)
            self.play(Flash(bars[-1], color=PURPLE, line_length=0.3), Write(bh_form))
            self.wait(2)
            self.play(*[FadeOut(m) for m in [bars, threshold_line, thresh_label, bh_form]])

            # -- Convergence event ---------------------------------------------
            conv_title = Text("CONVERGENCE EVENT", font_size=36, color=GOLD)
            self.play(Write(conv_title))

            circles = VGroup(*[
                Circle(radius=0.6, fill_color=BLACK, fill_opacity=1,
                       stroke_color=color, stroke_width=5).shift(pos)
                for color, pos in [(BLUE, LEFT*2.5), (GREEN, ORIGIN), (PURPLE, RIGHT*2.5)]
            ])
            labels = VGroup(*[
                Text(name, color=WHITE, font_size=20).move_to(c)
                for name, c in [("ES", LEFT*2.5), ("NQ", ORIGIN), ("YM", RIGHT*2.5)]
            ])

            conv_title.shift(UP*2.5)
            self.play(*[FadeIn(c) for c in circles], *[Write(l) for l in labels])

            star = Star(n=6, outer_radius=0.8, color=GOLD, fill_opacity=0.8)
            self.play(GrowFromCenter(star), run_time=1.5)
            conv_text = Text("All 3 BH active --> ENTER POSITION",
                             font_size=22, color=GOLD).shift(DOWN*2.5)
            self.play(Write(conv_text))
            self.wait(3)

            # -- End -----------------------------------------------------------
            end = Text("SRFM: Where Physics Meets Markets", font_size=28, color=BLUE)
            self.play(*[FadeOut(m) for m in self.mobjects])
            self.play(Write(end))
            self.wait(2)

except ImportError:
    pass
'''

# Execute the manim scene definition so the class exists in this module's namespace
exec(MANIM_SCENE_CODE, globals())


def generate_svg_storyboard():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    frames = [
        ("frame_01_metric.svg",        frame_01_metric()),
        ("frame_02_beta.svg",          frame_02_beta()),
        ("frame_03_classification.svg",frame_03_classification()),
        ("frame_04_mass.svg",          frame_04_mass()),
        ("frame_05_formation.svg",     frame_05_formation()),
        ("frame_06_convergence.svg",   frame_06_convergence()),
    ]

    frame_paths = []
    for fname, content in frames:
        path = os.path.join(FRAMES_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        frame_paths.append(path)
        print(f"  Saved: {path}")

    # Storyboard markdown
    storyboard_lines = [
        "# SRFM Manim Storyboard",
        "",
        "6 SVG frames explaining SRFM physics.",
        "",
        "## Frames",
        "",
    ]
    descriptions = [
        "Minkowski metric equation and light cone diagram",
        "Beta calculation: |Δp/p|/cf with number line",
        "TIMELIKE vs SPACELIKE classification on light cone",
        "BH mass accrual as bar chart with threshold line",
        "BH formation: mass crosses 1.5 threshold",
        "Convergence event: 3 instruments all BH active",
    ]
    for i, (fname, desc) in enumerate(zip([f[0] for f in frames], descriptions), 1):
        storyboard_lines.append(f"### Frame {i}: {fname}")
        storyboard_lines.append(f"{desc}")
        storyboard_lines.append(f"![Frame {i}](manim_frames/{fname})")
        storyboard_lines.append("")

    storyboard_lines.append("## Manim Usage")
    storyboard_lines.append("")
    storyboard_lines.append("```bash")
    storyboard_lines.append("pip install manim")
    storyboard_lines.append("manim -pql tools/manim_srfm.py SRFM   # low quality preview")
    storyboard_lines.append("manim -pqh tools/manim_srfm.py SRFM   # high quality")
    storyboard_lines.append("```")

    md_path = os.path.join(RESULTS_DIR, "manim_storyboard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(storyboard_lines))
    print(f"  Saved: {md_path}")
    return frame_paths, md_path


def main():
    parser = argparse.ArgumentParser(description="SRFM Manim animation / SVG storyboard")
    parser.add_argument("--svg-only", action="store_true", help="Force SVG storyboard even if manim available")
    args = parser.parse_args()

    manim_available = try_import_manim()

    print("SRFM MANIM ANIMATION / STORYBOARD")
    print("=" * 38)

    if manim_available and not args.svg_only:
        print("  manim is available.")
        print("  Run: manim -pql tools/manim_srfm.py SRFM   (for video)")
        print("  Generating SVG storyboard in addition...")
    else:
        if not manim_available:
            print("  manim not installed (pip install manim)")
        print("  Generating SVG storyboard...")

    frame_paths, md_path = generate_svg_storyboard()

    print(f"\nStoryboard complete: {len(frame_paths)} SVG frames")
    print(f"  Frames dir:  {FRAMES_DIR}")
    print(f"  Storyboard:  {md_path}")

    if not manim_available:
        print("\nTo render the full animation:")
        print("  pip install manim")
        print("  manim -pql tools/manim_srfm.py SRFM")


if __name__ == "__main__":
    main()
