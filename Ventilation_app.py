import dash
from dash import dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import base64
import io
from functools import lru_cache
from PIL import Image

# --- CONFIGURATION CONSTANTS ---
CONFIG = {
    "app": {
        "host": "127.0.0.1",
        "port": 8050,
        "debug": False
    },
    "colors": {
        "primary": "#2c3e50",
        "secondary": "#3498db",
        "accent": "#e74c3c",
        "text_secondary": "#555",
        "background": "#f8fafc",
        "panel_bg": "#ffffff",
        "border": "#dee2e6",
        "wall_bg": [200, 200, 200],  # Converted to list for vectorization
        "brick": [140, 70, 70],
        "window_frame": [240, 240, 240],
        "glass_light": [173, 216, 230],
        "glass_dark": [100, 149, 237],
        "glass_pivot": [70, 130, 180],
        "wood_base": [210, 180, 140],
        "hinge": [80, 80, 80]
    },
    "layout": {
        "wall_width": 500,
        "wall_height": 400,
        "frame_thickness": 10
    },
    "physics": {
        "gravity": 9.81,
        "air_density": 1.2,
        "discharge_coeff": 0.6,
        "temp_diff": 5.0,
        "temp_avg": 293.0,
        "wind_speed": 3.0,
        "room_volume": 50.0,
        "wall_height_m": 3.0,
        "wall_width_m": 3.0,
        "max_window_ratio": 0.6
    },
    "window_types": {
        1: {"name": "Side-hung window", "flow": 0.9, "wind": 0.8, "stack": 0.7},
        2: {"name": "Top-hung window", "flow": 0.8, "wind": 0.7, "stack": 0.9},
        3: {"name": "Bottom-hung window", "flow": 0.7, "wind": 0.6, "stack": 0.5},
        4: {"name": "Sliding window", "flow": 0.6, "wind": 0.5, "stack": 0.4},
        5: {"name": "Pivot window", "flow": 0.85, "wind": 0.75, "stack": 0.8}
    },
    "heatmap": {
        "size_resolution": 50,
        "position_resolution": 30,
        "colorscale": "Jet"
    },
    "brick": {
        "row_height": 20,
        "spacing": 60,
        "width": 50,
        "height": 15,
        "offset": 30
    },
    "branding": {
        "sintef_logo_url": "/assets/sintef_logo.png",
        "show_logo": True
    },
}

# --- CORE CALCULATION FUNCTIONS ---

def calculate_air_exchange_rate(window_type, width, height, angle, x_pos, y_pos):
    """Return ACH for scalar values or NumPy arrays of window parameters."""
    p = CONFIG["physics"]
    w_config = CONFIG["window_types"][window_type]

    width, height = np.asarray(width), np.asarray(height)
    angle, x_pos, y_pos = np.asarray(angle), np.asarray(x_pos), np.asarray(y_pos)
    win_width = (width / 100) * p["max_window_ratio"] * p["wall_width_m"]
    win_height = (height / 100) * p["max_window_ratio"] * p["wall_height_m"]
    win_area = win_width * win_height
    
    win_y_center = (y_pos / 8.0) * p["wall_height_m"]
    effective_height = win_y_center + (win_height / 2.0)
    
    opening_factor = angle / 90.0
    effective_area = win_area * opening_factor * w_config["flow"]
    
    type_corrections = {2: 0.9, 3: 0.8, 5: 0.95}
    if window_type in type_corrections:
        effective_area *= type_corrections[window_type]
    
    stack_flow = (p["discharge_coeff"] * effective_area *
                  np.sqrt(np.maximum(0, 2 * p["gravity"] * effective_height *
                         (p["temp_diff"] / p["temp_avg"]))))
    
    wind_position_factor = 1.0 - (x_pos / 20.0)
    wind_flow = (p["discharge_coeff"] * effective_area * p["wind_speed"] *
                 w_config["wind"] * wind_position_factor)
    
    combined_flow = np.sqrt(stack_flow**2 + wind_flow**2)
    ach = (combined_flow * 3600) / p["room_volume"]
    
    angle_correction = opening_factor ** 0.7
    height_factor = 0.5 + (effective_height / p["wall_height_m"]) * 0.8
    
    return ach * angle_correction * height_factor * w_config["stack"]

# --- VECTORIZED IMAGE GENERATION FUNCTIONS ---

def create_wall_image(window_type, width, height, angle, x_pos, y_pos):
    img = _create_brick_wall()
    
    win_width = int((width / 100) * CONFIG["layout"]["wall_width"] * CONFIG["physics"]["max_window_ratio"])
    win_height = int((height / 100) * CONFIG["layout"]["wall_height"] * CONFIG["physics"]["max_window_ratio"])
    
    frame_t = CONFIG["layout"]["frame_thickness"]
    max_x = CONFIG["layout"]["wall_width"] - win_width - frame_t
    max_y = CONFIG["layout"]["wall_height"] - win_height - frame_t
    
    win_x = max(frame_t, min(int(x_pos * (max_x / 10)), max_x))
    win_y = max(frame_t, min(int(y_pos * (max_y / 8)), max_y))
    
    _draw_window_frame(img, win_x, win_y, win_width, win_height)
    _draw_window_glass(img, win_x, win_y, win_width, win_height)
    _draw_window_opening(img, win_x, win_y, win_width, win_height, window_type, angle)
    
    return _add_window_textures(img, win_x, win_y, win_width, win_height, window_type, angle)

@lru_cache(maxsize=1)
def _brick_wall_template():
    l = CONFIG["layout"]
    b = CONFIG["brick"]
    img = np.full((l["wall_height"], l["wall_width"], 3), CONFIG["colors"]["wall_bg"], dtype=np.uint8)
    
    for y in range(0, l["wall_height"], b["row_height"]):
        offset = b["offset"] if (y // b["row_height"]) % 2 else 0
        for x in range(offset, l["wall_width"], b["spacing"]):
            brick_w = min(b["width"], l["wall_width"] - x)
            brick_h = min(b["height"], l["wall_height"] - y)
            if brick_w > 0 and brick_h > 0:
                img[y:y+brick_h, x:x+brick_w] = CONFIG["colors"]["brick"]
    return img


def _create_brick_wall():
    """Copy the immutable wall template before drawing a user-specific window."""
    return _brick_wall_template().copy()

def _draw_window_frame(img, x, y, w, h):
    t = CONFIG["layout"]["frame_thickness"]
    img[y-t:y+h+t, x-t:x+w+t] = CONFIG["colors"]["window_frame"]

def _draw_window_glass(img, x, y, w, h):
    img[y:y+h, x:x+w] = CONFIG["colors"]["glass_light"]

def _draw_window_opening(img, x, y, w, h, window_type, angle):
    factor = angle / 90.0
    c_dark = CONFIG["colors"]["glass_dark"]
    
    if window_type == 1:
        img[y:y+h, x:x+int(w*factor)] = c_dark        # Removed *0.5
    elif window_type == 2:
        img[y:y+int(h*factor), x:x+w] = c_dark        # Removed *0.5
    elif window_type == 3:
        img[y+h-int(h*factor):y+h, x:x+w] = c_dark    # Removed *0.5
    elif window_type == 4:
        img[y:y+h, x+w-int(w*factor*0.5):x+w] = c_dark # Kept *0.5 (sliding)
    elif window_type == 5:
        pivot_y = y + h // 2
        open_h = int(h * factor * 0.3)
        img[pivot_y-open_h:pivot_y, x:x+w] = c_dark
        img[pivot_y:pivot_y+open_h, x:x+w] = CONFIG["colors"]["glass_pivot"]


def _add_window_textures(img, x, y, w, h, window_type, angle):
    t_img = img.copy()
    factor = angle / 90.0
    c = CONFIG["colors"]
    
    _apply_wood_frame_texture(t_img, x, y, w, h)
    
    if w <= 0 or h <= 0: return t_img
    
    # Convert colors to numpy arrays for reliable math operations
    c_light = np.array(c["glass_light"])
    c_dark = np.array(c["glass_dark"])
    
    if window_type == 1:  # Side-hung
        open_w = int(w * factor )
        if open_w < w:  # Fixed Glass
            r_grid, c_grid = np.ogrid[y:y+h, x+open_w:x+w]
            refl = np.sin(c_grid * 0.1 + r_grid * 0.1) * 15 + 15
            grad = (r_grid - y) / max(h, 1) * 20
            effect = (refl + grad)[..., np.newaxis] # Add Z-dimension
            t_img[y:y+h, x+open_w:x+w] = np.clip(c_light + effect, 0, 255)
            
        if open_w > 0:  # Moving Glass
            r_grid, c_grid = np.ogrid[y:y+h, x:x+open_w]
            shadow = 30 * (1 - (c_grid - x) / max(open_w, 1))
            refl = np.sin((c_grid - x) * 0.2 + r_grid * 0.1) * 10
            effect = (refl - shadow)[..., np.newaxis]
            t_img[y:y+h, x:x+open_w] = np.clip(c_dark + effect, 0, 255)
            t_img[y:y+h, max(x, x+open_w-1):min(x+w, x+open_w+1)] = c["hinge"]

    elif window_type == 2:  # Top-hung
        open_h = int(h * factor )
        if open_h < h:  # Fixed
            r_grid, c_grid = np.ogrid[y+open_h:y+h, x:x+w]
            refl = np.sin(c_grid*0.1 + r_grid*0.1) * 15 + 15
            grad = (r_grid - y) / max(h, 1) * 20
            effect = (refl + grad)[..., np.newaxis]
            t_img[y+open_h:y+h, x:x+w] = np.clip(c_light + effect, 0, 255)
            
        if open_h > 0:  # Moving
            r_grid, c_grid = np.ogrid[y:y+open_h, x:x+w]
            shadow = 30 * (1 - (r_grid - y) / max(open_h, 1))
            refl = np.sin(c_grid*0.1 + (r_grid - y)*0.2) * 10
            effect = (refl - shadow)[..., np.newaxis]
            t_img[y:y+open_h, x:x+w] = np.clip(c_dark + effect, 0, 255)
            t_img[max(y, y+open_h-1):min(y+h, y+open_h+1), x:x+w] = c["hinge"]

    elif window_type == 3:  # Bottom-hung
        open_h = int(h * factor )
        if open_h < h:  # Fixed
            r_grid, c_grid = np.ogrid[y:y+h-open_h, x:x+w]
            refl = np.sin(c_grid*0.1 + r_grid*0.1) * 15 + 15
            grad = (r_grid - y) / max(h, 1) * 20
            effect = (refl + grad)[..., np.newaxis]
            t_img[y:y+h-open_h, x:x+w] = np.clip(c_light + effect, 0, 255)
            
        if open_h > 0:  # Moving
            r_grid, c_grid = np.ogrid[y+h-open_h:y+h, x:x+w]
            shadow = 30 * ((r_grid - (y+h-open_h)) / max(open_h, 1))
            refl = np.sin(c_grid*0.1 + (r_grid - (y+h-open_h))*0.2) * 10
            effect = (refl - shadow)[..., np.newaxis]
            t_img[y+h-open_h:y+h, x:x+w] = np.clip(c_dark + effect, 0, 255)
            t_img[max(y, y+h-open_h-1):min(y+h, y+h-open_h+1), x:x+w] = c["hinge"]

    elif window_type == 4:  # Sliding
        open_w = int(w * factor * 0.5)
        if open_w < w:  # Fixed
            r_grid, c_grid = np.ogrid[y:y+h, x:x+w-open_w]
            refl = np.sin(c_grid*0.1 + r_grid*0.1) * 15 + 15
            grad = (r_grid - y) / max(h, 1) * 20
            effect = (refl + grad)[..., np.newaxis]
            t_img[y:y+h, x:x+w-open_w] = np.clip(c_light + effect, 0, 255)
            
        if open_w > 0:  # Moving
            r_grid, c_grid = np.ogrid[y:y+h, x+w-open_w:x+w]
            shadow = 30 * ((c_grid - (x+w-open_w)) / max(open_w, 1))
            refl = np.sin((c_grid - (x+w-open_w))*0.2 + r_grid*0.1) * 10
            effect = (refl - shadow)[..., np.newaxis]
            t_img[y:y+h, x+w-open_w:x+w] = np.clip(c_dark + effect, 0, 255)
            t_img[y:y+h, max(x, x+w-open_w-1):min(x+w, x+w-open_w+1)] = c["hinge"]

    elif window_type == 5:  # Pivot
        pivot_y = y + h // 2
        open_h = int(h * factor * 0.3)
        c_pivot = np.array(c["glass_pivot"])
        
        if open_h < h//2:  # Top fixed
            r_grid, c_grid = np.ogrid[y:pivot_y-open_h, x:x+w]
            effect = (np.sin(c_grid*0.1 + r_grid*0.1)*15+15 + (r_grid-y)/max(h,1)*20)[..., np.newaxis]
            t_img[y:pivot_y-open_h, x:x+w] = np.clip(c_light + effect, 0, 255)
            
        if open_h > 0:  # Top moving
            r_grid, c_grid = np.ogrid[pivot_y-open_h:pivot_y, x:x+w]
            effect = (np.sin(c_grid*0.1 + (pivot_y-r_grid)*0.2)*10 - 30*(1-(pivot_y-r_grid)/max(open_h,1)))[..., np.newaxis]
            t_img[pivot_y-open_h:pivot_y, x:x+w] = np.clip(c_dark + effect, 0, 255)
            
            # Bottom moving
            r_grid, c_grid = np.ogrid[pivot_y:pivot_y+open_h, x:x+w]
            effect = (np.sin(c_grid*0.1 + (r_grid-pivot_y)*0.2)*10 - 30*((r_grid-pivot_y)/max(open_h,1)))[..., np.newaxis]
            t_img[pivot_y:pivot_y+open_h, x:x+w] = np.clip(c_pivot + effect, 0, 255)
            
        if open_h < h - h//2:  # Bottom fixed
            r_grid, c_grid = np.ogrid[pivot_y+open_h:y+h, x:x+w]
            effect = (np.sin(c_grid*0.1 + r_grid*0.1)*15+15 + (r_grid-y)/max(h,1)*20)[..., np.newaxis]
            t_img[pivot_y+open_h:y+h, x:x+w] = np.clip(c_light + effect, 0, 255)
            
        t_img[max(y, pivot_y-1):min(y+h, pivot_y+1), x:x+w] = c["hinge"]

    return t_img


def _apply_wood_frame_texture(img, x, y, w, h):
    t = CONFIG["layout"]["frame_thickness"]
    base = np.array(CONFIG["colors"]["wood_base"])
    areas = [
        (y-t, y+h+t, x-t, x),        # Left
        (y-t, y+h+t, x+w, x+w+t),    # Right  
        (y-t, y, x-t, x+w+t),        # Top
        (y+h, y+h+t, x-t, x+w+t)     # Bottom
    ]
    
    for ys, ye, xs, xe in areas:
        ys, ye = max(0, ys), min(img.shape[0], ye)
        xs, xe = max(0, xs), min(img.shape[1], xe)
        if ye > ys and xe > xs:
            # FIX: Reshape to (N, 1, 1) so it broadcasts across width (columns) and RGB channels
            row_idx = np.arange(ys, ye).reshape(-1, 1, 1)
            grain = np.sin(row_idx * 0.2) * 20 + 10
            img[ys:ye, xs:xe] = np.clip(base + grain, 0, 255)

# --- HEATMAP GENERATION ---

def create_size_heatmap(window_type, width, height, angle, x_pos, y_pos):
    res = CONFIG["heatmap"]["size_resolution"]
    x_vals = np.linspace(1, 100, res)
    y_vals = np.linspace(1, 100, res)
    widths, heights = np.meshgrid(x_vals, y_vals)
    z_vals = calculate_air_exchange_rate(window_type, widths, heights, angle, x_pos, y_pos)
    return _create_heatmap_figure(x_vals, y_vals, z_vals, width, height, 'Air Exchange by Size', 'Width (%)', 'Height (%)')

def create_position_heatmap(window_type, width, height, angle, x_pos, y_pos):
    res = CONFIG["heatmap"]["position_resolution"]
    x_vals, y_vals = np.linspace(0, 10, res), np.linspace(0, 8, res)
    x_positions, y_positions = np.meshgrid(x_vals, y_vals)
    z_vals = calculate_air_exchange_rate(window_type, width, height, angle, x_positions, y_positions)
    return _create_heatmap_figure(x_vals, y_vals, z_vals, x_pos, y_pos, 'Air Exchange by Position', 'Horizontal', 'Vertical')

def _create_heatmap_figure(x_vals, y_vals, z_vals, cx, cy, title, xt, yt):
    fig = go.Figure(data=[
        go.Heatmap(z=z_vals, x=x_vals, y=y_vals, colorscale=CONFIG["heatmap"]["colorscale"], 
                   colorbar=dict(title='ACH [1/h]')),
        go.Contour(z=z_vals, x=x_vals, y=y_vals, colorscale=CONFIG["heatmap"]["colorscale"], 
                   showscale=False, contours=dict(showlabels=True, labelfont=dict(color='white')), 
                   line=dict(width=0.5, color='rgba(255,255,255,0.8)')),
        go.Scatter(x=[cx], y=[cy], mode='markers+text', text=['Current'], textposition="top center",
                   marker=dict(color='white', size=10, line=dict(color='black', width=2)))
    ])
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, height=350, margin=dict(l=40, r=20, t=40, b=40))
    return fig

# --- DASH APPLICATION ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME])
server = app.server

def create_slider(id, label, min_val, max_val, step, val, marks):
    return dbc.Row([
        dbc.Col(html.Label(label, className="fw-bold"), width=4),
        dbc.Col(dcc.Slider(
            id=id, min=min_val, max=max_val, step=step, value=val, marks=marks,
            updatemode='mouseup', tooltip={"placement": "bottom", "always_visible": True}
        ), width=8)
    ], className="mb-3 align-items-center")

# UI Layout Using Dash Bootstrap Components
app.layout = dbc.Container([
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div("NATURAL VENTILATION · DESIGN TOOL", className="eyebrow"),
            html.H1("Ventilation Design Explorer", className="app-title"),
            html.P("Explore how window geometry, opening angle and placement affect air exchange.", className="app-subtitle")
        ], md=9),
        dbc.Col(html.Img(src=CONFIG["branding"]["sintef_logo_url"], className="brand-logo") if CONFIG["branding"]["show_logo"] else None, md=3, className="brand-logo-wrap")
    ], className="app-header"),
    
    # Credits & Assumptions Card
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5("Scientific Assumptions", className="card-title text-secondary border-bottom pb-2"),
                html.Ul([
                    html.Li("Wind Direction: Left to right (better on left side)"),
                    html.Li("Room Volume: 50 m³ | Temp Diff: 5°C | Wind Speed: 3 m/s"),
                    html.Li("Wall Dimensions: 3m × 3m")
                ], className="small mb-0")
            ], width=8),
            dbc.Col([
                html.P([
                    html.Strong("Made by Timothée ALLIO"), html.Br(),
                    html.Span("Based on SINTEF research (Kleiven & Hestnes) and hybrid ventilation systems.", className="small text-muted")
                ], className="text-end mb-0")
            ], width=4)
        ])
    ]), className="intro-card mb-4"),

    # Main Interface
    dbc.Row([
        # Left Panel: Controls & Metrics
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Window Parameters", className="mb-4"),
                dbc.Row([
                    dbc.Col(html.Label("Window Type", className="fw-bold"), width=4),
                    dbc.Col(dcc.Dropdown(
                        id='window-type-dropdown',
                        options=[{'label': v["name"], 'value': k} for k, v in CONFIG["window_types"].items()],
                        value=1, clearable=False
                    ), width=8)
                ], className="mb-3 align-items-center"),
                
                create_slider('window-width-slider', 'Width (%)', 0, 100, 5, 50, {0: '0', 50: '50', 100: '100'}),
                create_slider('window-height-slider', 'Height (%)', 0, 100, 5, 50, {0: '0', 50: '50', 100: '100'}),
                create_slider('window-angle-slider', 'Angle (°)', 0, 90, 5, 45, {0: '0', 45: '45', 90: '90'}),
                create_slider('horizontal-position-slider', 'X Position', 0, 10, 1, 5, {0: '0', 5: '5', 10: '10'}),
                create_slider('vertical-position-slider', 'Y Position', 0, 8, 1, 4, {0: '0', 4: '4', 8: '8'}),
                
                html.Hr(),
                html.Div(id='metrics-display', className="text-center p-3 bg-light rounded border")
            ]), className="control-card h-100")
        ], md=5),
        
        # Right Panel: Visualization & Analytics
        dbc.Col([
            dbc.Card(dbc.CardBody([
                dcc.Loading(
                    id="loading-image", type="default",
                    children=html.Img(id='building-image', className="building-image")
                ),
                dbc.Tabs(id='analytics-tabs', active_tab='size-map', className="analytics-tabs", children=[
                    dbc.Tab(dcc.Loading(dcc.Graph(id='size-heatmap', config={'displayModeBar': False}, responsive=True)), label="Size Effect Map", tab_id='size-map', className="pt-3"),
                    dbc.Tab(dcc.Loading(dcc.Graph(id='position-heatmap', config={'displayModeBar': False}, responsive=True)), label="Position Effect Map", tab_id='position-map', className="pt-3")
                ])
            ]), className="visualization-card")
        ], md=7)
    ])
], fluid=True, className="app-shell")

# --- CALLBACKS ---

@app.callback(
    [Output('building-image', 'src'),
     Output('metrics-display', 'children')],
    [Input('window-type-dropdown', 'value'),
     Input('window-width-slider', 'value'),
     Input('window-height-slider', 'value'),
     Input('window-angle-slider', 'value'),
     Input('horizontal-position-slider', 'value'),
     Input('vertical-position-slider', 'value')]
)
def update_dashboard(window_type, width, height, angle, x_pos, y_pos):
    # Vectorized Image Generation
    img = create_wall_image(window_type, width, height, angle, x_pos, y_pos)
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    img_src = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    
    # Core Math Calculation
    ach = calculate_air_exchange_rate(window_type, width, height, angle, x_pos, y_pos)
    
    p = CONFIG["physics"]
    window_area = ((width / 100) * p["max_window_ratio"] * p["wall_width_m"] *
                   (height / 100) * p["max_window_ratio"] * p["wall_height_m"])
    metrics = html.Div([
        html.Span("Estimated air exchange", className="metric-label"),
        html.Div([html.Span(f"{ach:.2f}", className="metric-value"), html.Span(" ACH", className="metric-unit")]),
        html.Div([
            html.Span(f"Window area {window_area:.2f} m²"),
            html.Span("•"),
            html.Span(f"Position {x_pos} / {y_pos}")
        ], className="metric-details")
    ])
    
    return img_src, metrics


@app.callback(
    Output('size-heatmap', 'figure'),
    [Input('analytics-tabs', 'active_tab'),
     Input('window-type-dropdown', 'value'),
     Input('window-width-slider', 'value'),
     Input('window-height-slider', 'value'),
     Input('window-angle-slider', 'value'),
     Input('horizontal-position-slider', 'value'),
     Input('vertical-position-slider', 'value')]
)
def update_size_heatmap(active_tab, window_type, width, height, angle, x_pos, y_pos):
    if active_tab != 'size-map':
        return no_update
    return create_size_heatmap(window_type, width, height, angle, x_pos, y_pos)


@app.callback(
    Output('position-heatmap', 'figure'),
    [Input('analytics-tabs', 'active_tab'),
     Input('window-type-dropdown', 'value'),
     Input('window-width-slider', 'value'),
     Input('window-height-slider', 'value'),
     Input('window-angle-slider', 'value'),
     Input('horizontal-position-slider', 'value'),
     Input('vertical-position-slider', 'value')]
)
def update_position_heatmap(active_tab, window_type, width, height, angle, x_pos, y_pos):
    if active_tab != 'position-map':
        return no_update
    return create_position_heatmap(window_type, width, height, angle, x_pos, y_pos)

if __name__ == '__main__':
    app.run(host=CONFIG["app"]["host"], port=CONFIG["app"]["port"], debug=CONFIG["app"]["debug"])
