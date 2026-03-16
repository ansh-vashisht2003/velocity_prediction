import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import tree
import numpy as np
from models.model_utils import predict_velocity
import datetime
import os
import serial
import threading

# dataset
df = pd.read_csv("data/fake_gas_gun_dataset.csv")

projectile_types = sorted(df["Projectile Type"].dropna().unique())
shapes = sorted(df["Shape"].dropna().unique())
materials = sorted(df["Material"].dropna().unique())

calibres = ["40 mm", "29 mm"]


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("BALLISTICS AI — TACTICAL SIMULATOR")
        root.geometry("1600x920")

        # ── SOFTER TACTICAL PALETTE (easy on eyes) ───────────────────────────
        self.C = {
            'bg':           '#1c2030',   # softer dark blue-grey base
            'surface':      '#242a3a',   # card surface
            'surface2':     '#2a3248',   # elevated surface
            'border':       '#364256',   # visible border
            'border_bright':'#4a6080',   # hover/active border
            'accent':       '#5bc8e8',   # softer sky blue
            'accent2':      '#4dd9a0',   # softer mint green
            'danger':       '#e86070',   # softer red
            'warn':         '#e8b050',   # softer amber
            'text':         '#dde6f0',   # soft white
            'text_dim':     '#7a96b0',   # readable muted text
            'text_mid':     '#a0bcd0',   # mid text
            'header_line':  '#5bc8e8',   # top accent line
            'input_bg':     '#1e2638',   # input background
            'row_even':     '#242a3a',
            'row_odd':      '#202636',
            'sel':          '#2a4a6e',
        }

        root.configure(bg=self.C['bg'])
        self._setup_fonts()
        self._setup_styles()

        # Combobox dropdown colors
        root.option_add('*TCombobox*Listbox.background', self.C['input_bg'])
        root.option_add('*TCombobox*Listbox.foreground', self.C['text'])
        root.option_add('*TCombobox*Listbox.selectBackground', self.C['accent'])
        root.option_add('*TCombobox*Listbox.selectForeground', '#000000')
        root.option_add('*TCombobox*Listbox.font', ('Consolas', 11))

        self._build_header()
        self._build_notebook()

        self.predicted_velocity = 0
        self.actual_velocity = 0
        self.shot_no = 1
        
       
        try:
            self.ser = serial.Serial("COM5", 9600, timeout=1)
            print("Arduino Connected")
            threading.Thread(target=self.read_arduino, daemon=True).start()
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")

    # ── FONTS ─────────────────────────────────────────────────────────────────
    def _setup_fonts(self):
        self.font_title    = ('Courier New', 15, 'bold')
        self.font_label    = ('Courier New', 11)
        self.font_mono     = ('Consolas', 11)
        self.font_value    = ('Courier New', 24, 'bold')
        self.font_value_sm = ('Courier New', 16, 'bold')
        self.font_tab      = ('Courier New', 11, 'bold')
        self.font_btn      = ('Courier New', 12, 'bold')
        self.font_status   = ('Consolas', 10)
        self.font_head     = ('Courier New', 20, 'bold')

    # ── STYLES ────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use('default')

        # Notebook
        s.configure('T.TNotebook',
                    background=self.C['bg'],
                    borderwidth=0,
                    tabmargins=[0, 0, 0, 0])
        s.configure('T.TNotebook.Tab',
                    background=self.C['surface'],
                    foreground=self.C['text_dim'],
                    padding=[18, 9],
                    font=self.font_tab,
                    borderwidth=0,
                    focuscolor=self.C['bg'])
        s.map('T.TNotebook.Tab',
              background=[('selected', self.C['bg'])],
              foreground=[('selected', self.C['accent'])],
              expand=[('selected', [0, 0, 0, 2])])

        # Combobox
        s.configure('T.TCombobox',
                    fieldbackground=self.C['input_bg'],
                    foreground=self.C['text'],
                    arrowcolor=self.C['accent'],
                    borderwidth=1,
                    relief='flat',
                    font=self.font_mono)
        s.map('T.TCombobox',
              fieldbackground=[('readonly', self.C['input_bg'])],
              foreground=[('readonly', self.C['text'])])

        # Treeview — tactical grid
        s.configure('T.Treeview',
                    background=self.C['surface'],
                    foreground=self.C['text_mid'],
                    fieldbackground=self.C['surface'],
                    borderwidth=0,
                    rowheight=32,
                    font=self.font_mono)
        s.configure('T.Treeview.Heading',
                    background=self.C['surface2'],
                    foreground=self.C['accent'],
                    relief='flat',
                    font=('Courier New', 11, 'bold'))
        s.map('T.Treeview',
              background=[('selected', self.C['sel'])],
              foreground=[('selected', self.C['accent'])])

        # Scrollbar
        s.configure('T.Vertical.TScrollbar',
                    troughcolor=self.C['bg'],
                    background=self.C['border'],
                    borderwidth=0,
                    arrowsize=12)
        s.configure('T.Horizontal.TScrollbar',
                    troughcolor=self.C['bg'],
                    background=self.C['border'],
                    borderwidth=0,
                    arrowsize=12)

    # ── HEADER ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.C['surface2'], height=52)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        # Top accent line
        line = tk.Frame(self.root, bg=self.C['accent'], height=2)
        line.pack(fill='x', side='top')
        line.lower(hdr)

        accent_bar = tk.Frame(hdr, bg=self.C['accent'], width=4)
        accent_bar.pack(side='left', fill='y', padx=(16, 12))

        tk.Label(hdr, text='◈  BALLISTICS AI', bg=self.C['surface2'],
                 fg=self.C['accent'], font=('Courier New', 14, 'bold')).pack(side='left')
        tk.Label(hdr, text='TACTICAL SIMULATOR v2.0', bg=self.C['surface2'],
                 fg=self.C['text_dim'], font=self.font_status).pack(side='left', padx=12)

        # Right status badge
        badge = tk.Frame(hdr, bg=self.C['bg'], padx=10, pady=4)
        badge.pack(side='right', padx=16, pady=8)
        tk.Label(badge, text='● SYSTEM READY', bg=self.C['bg'],
                 fg=self.C['accent2'], font=self.font_status).pack()

    # ── NOTEBOOK ──────────────────────────────────────────────────────────────
    def _build_notebook(self):
        wrapper = tk.Frame(self.root, bg=self.C['bg'])
        wrapper.pack(fill='both', expand=True)

        nb = ttk.Notebook(wrapper, style='T.TNotebook')
        tabs = []
        tab_names = [
        "  🚀  VELOCITY  ",
        "  ⚖   POWDER    ",
        "  ⚙   PHYSICS   ",
        "  📊  GRAPHS    ",
        "  🎯  ACCURACY  ",
        "  📁  DATASET   ",
        "  📄  REPORT    "
        ]
        
        for name in tab_names:
            f = tk.Frame(nb, bg=self.C['bg'])
            tabs.append(f)
            nb.add(f, text=name)

        nb.pack(expand=True, fill='both', padx=0, pady=0)

        self.build_tab1(tabs[0])
        self.build_tab2(tabs[1])
        self.build_tab3(tabs[2])
        self.build_tab4(tabs[3])
        self.build_tab5(tabs[4])
        self.build_tab6(tabs[5])
        self.build_tab7(tabs[6])

    # ── CARD ──────────────────────────────────────────────────────────────────
    def create_card(self, parent, title, accent_color):
        outer = tk.Frame(parent, bg=self.C['border'], padx=1, pady=1)
        card  = tk.Frame(outer, bg=self.C['surface'])
        card.pack(fill='both', expand=True)

        # Coloured top bar (thin)
        tk.Frame(card, bg=accent_color, height=3).pack(fill='x')

        # Title row
        title_row = tk.Frame(card, bg=self.C['surface2'])
        title_row.pack(fill='x')
        tk.Label(title_row, text=f'  {title}',
                 bg=self.C['surface2'], fg=accent_color,
                 font=('Courier New', 12, 'bold'),
                 anchor='w').pack(side='left', pady=8)
        # Decorative corner glyph
        tk.Label(title_row, text='◫  ', bg=self.C['surface2'],
                 fg=self.C['text_dim'], font=self.font_status).pack(side='right')

        return outer, card

    # ── INPUT FIELD ───────────────────────────────────────────────────────────
    def create_input_field(self, parent, label, row, col,
                           is_combo=False, values=None, default=''):
        c = self.C

        tk.Label(parent, text=label.upper(),
                 bg=c['surface'], fg=c['text_dim'],
                 font=('Courier New', 10),
                 anchor='w').grid(row=row, column=col*2,
                                   sticky='w', pady=6, padx=(10, 8))

        if is_combo:
            w = ttk.Combobox(parent, values=values, width=18,
                             style='T.TCombobox', font=self.font_mono)
            if default:
                w.set(default)
        else:
            w = tk.Entry(parent,
                         bg=c['input_bg'], fg=c['text'],
                         insertbackground=c['accent'],
                         relief='flat', bd=0, width=20,
                         font=self.font_mono,
                         highlightbackground=c['border'],
                         highlightcolor=c['accent'],
                         highlightthickness=1)
            if default:
                w.insert(0, default)

        w.grid(row=row, column=col*2+1, sticky='w', padx=(0, 14), pady=5)
        return w

    # ── BUTTON ────────────────────────────────────────────────────────────────
    def create_button(self, parent, text, command, color):
        btn = tk.Button(
            parent, text=f'▶  {text}', command=command,
            bg=self.C['surface2'], fg=color,
            font=('Courier New', 12, 'bold'),
            relief='flat', bd=0, cursor='hand2',
            padx=30, pady=12,
            activebackground=self.C['border'],
            activeforeground=color,
            highlightbackground=color,
            highlightthickness=1
        )

        def on_enter(e):
            btn.config(bg=self.C['border'], fg='#ffffff')
        def on_leave(e):
            btn.config(bg=self.C['surface2'], fg=color)

        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        return btn

    # ── SEPARATOR ─────────────────────────────────────────────────────────────
    def _sep(self, parent, row, cols=4):
        tk.Frame(parent, bg=self.C['border'], height=1).grid(
            row=row, column=0, columnspan=cols, sticky='ew',
            padx=10, pady=4)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — VELOCITY
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab1(self, tab):
        left = tk.Frame(tab, bg=self.C['bg'])
        left.pack(side='left', fill='both', expand=True, padx=(8, 4), pady=8)

        right = tk.Frame(tab, bg=self.C['bg'])
        right.pack(side='right', fill='both', expand=True, padx=(4, 8), pady=8)

        # ── INPUT CARD
        outer, card = self.create_card(left, 'INPUT PARAMETERS', self.C['accent'])
        outer.pack(fill='both', expand=True)

        ct = tk.Frame(card, bg=self.C['surface'])
        ct.pack(fill='both', expand=True, padx=12, pady=10)

        r = 0
        self.calibre           = self.create_input_field(ct, 'Calibre',                r, 0, True, calibres, '29 mm'); r+=1
        self.projectile_type   = self.create_input_field(ct, 'Projectile Type',         r, 0, True, projectile_types); r+=1
        self.dimension         = self.create_input_field(ct, 'Projectile Dimension',    r, 0, default='10'); r+=1
        self.mass              = self.create_input_field(ct, 'Projectile Mass',         r, 0, default='5'); r+=1
        self.total_mass        = self.create_input_field(ct, 'Total Mass w/ Sabot',     r, 0, default='7'); r+=1
        self.powder            = self.create_input_field(ct, 'Powder Mass',             r, 0, default='3'); r+=1
        self.pressure          = self.create_input_field(ct, 'Petal Burst Pressure',    r, 0, default='80'); r+=1
        self.sensor_distance   = self.create_input_field(ct, 'Sensor Distance (m)',     r, 0, default='1'); r+=1

        self.shape   = self.create_input_field(ct, 'Shape',            0, 1, True, shapes)
        self.breadth = self.create_input_field(ct, 'Breadth',          1, 1, default='5')
        self.height  = self.create_input_field(ct, 'Height',           2, 1, default='8')
        self.material= self.create_input_field(ct, 'Material',         3, 1, True, materials)
        self.cdrag   = self.create_input_field(ct, 'Drag Coefficient', 4, 1, default='0.3')
        self.sa      = self.create_input_field(ct, 'Surface Area',     5, 1, default='300')
        self.vol     = self.create_input_field(ct, 'Volume',           6, 1, default='500')
        self.sabot   = self.create_input_field(ct, 'Sabot Length',     7, 1, default='40')

        self._sep(ct, 8, 4)

        btn_f = tk.Frame(ct, bg=self.C['surface'])
        btn_f.grid(row=9, column=0, columnspan=4, pady=12)
        self.predict_btn = self.create_button(btn_f, 'PREDICT VELOCITY', self.predict, self.C['accent'])
        self.predict_btn.pack()

        # ── RESULTS CARD
        outer2, card2 = self.create_card(right, 'VELOCITY RESULTS', self.C['accent2'])
        outer2.pack(fill='both', expand=True)

        rc = tk.Frame(card2, bg=self.C['surface'])
        rc.pack(fill='both', expand=True, padx=20, pady=20)

        # Big value displays
        for txt, attr, fg in [
            ('PREDICTED', 'predicted_label', self.C['accent']),
            ('ACTUAL',    'actual_label',    self.C['accent2']),
        ]:
            blk = tk.Frame(rc, bg=self.C['surface2'],
                           highlightbackground=self.C['border'],
                           highlightthickness=1)
            blk.pack(fill='x', pady=6, ipady=10)
            tk.Label(blk, text=txt, bg=self.C['surface2'],
                     fg=self.C['text_dim'], font=('Courier New', 10)).pack()
            lbl = tk.Label(blk, text='--- m/s', bg=self.C['surface2'],
                           fg=fg, font=('Courier New', 28, 'bold'))
            lbl.pack()
            setattr(self, attr, lbl)

        # Error block
        err_blk = tk.Frame(rc, bg=self.C['surface2'],
                           highlightbackground=self.C['border'],
                           highlightthickness=1)
        err_blk.pack(fill='x', pady=6, ipady=8)
        tk.Label(err_blk, text='ERROR', bg=self.C['surface2'],
                 fg=self.C['text_dim'], font=('Courier New', 10)).pack()
        self.error_label = tk.Label(err_blk, text='0.00 %',
                                    bg=self.C['surface2'],
                                    fg=self.C['warn'],
                                    font=('Courier New', 22, 'bold'))
        self.error_label.pack()

        # Shot history table
        tk.Frame(rc, bg=self.C['border'], height=1).pack(fill='x', pady=10)

        tk.Label(rc, text='▸ SHOT LOG', bg=self.C['surface'],
                 fg=self.C['text_dim'], font=('Courier New', 10, 'bold'),
                 anchor='w').pack(fill='x', pady=(0, 4))

        cols = ('Shot', 'Time', 'Velocity', 'Predicted', 'Error')
        self.shot_table = ttk.Treeview(rc, columns=cols, show='headings',
                                       height=7, style='T.Treeview')
        widths = [60,120,100,100,80]

        for i,c in enumerate(cols):
            self.shot_table.heading(c, text=c.upper())
            self.shot_table.column(c, width=widths[i], anchor='center')
        self.shot_table.pack(fill='both', expand=True)

    def update_actual_velocity(self, t1, t2):
        try:
            distance = float(self.sensor_distance.get())
            time_diff = (t2 - t1) / 1000
            velocity = distance / time_diff
            self.actual_velocity = velocity
            self.actual_label.config(text=f'{velocity:.2f} m/s')
            self.calculate_error()
        except:
            pass

    def calculate_error(self):

        import math

        if self.predicted_velocity == 0:
            error = float('nan')   # No prediction available
            error_display = "NaN"
            color = self.C['warn']
        else:
            error = abs(self.predicted_velocity - self.actual_velocity) / self.predicted_velocity * 100
            error_display = f"{error:.2f} %"
            color = self.C['accent2'] if error <= 10 else self.C['danger']

        self.error_label.config(text=error_display, fg=color)

        # Always save velocity to Excel
        self.save_excel(self.actual_velocity, error)
    def save_excel(self, velocity, error):
        file = 'shot_data.xlsx'
        now = datetime.datetime.now()
        data = {
            'Date': [now.strftime('%Y-%m-%d')],
            'Time': [now.strftime('%H:%M:%S')],
            'Shot No': [self.shot_no],
            'Velocity (m/s)': [velocity],
            'Predicted Velocity': [self.predicted_velocity],
            'Error %': [error]
        }
        df_new = pd.DataFrame(data)
        if os.path.exists(file):
            df_old = pd.read_excel(file)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        try:
            df_new.to_excel(file, index=False)
        except:
            print('Close Excel file to save data')

        time_now = now.strftime('%H:%M:%S')

        self.shot_table.insert('', 'end', values=(
            self.shot_no,
            time_now,
            round(velocity,2),
            round(self.predicted_velocity,2),
            f'{round(error,2)}%' if not pd.isna(error) else "NaN"
        ))
        self.shot_no += 1

    def load_shot_history(self):
        file = 'shot_data.xlsx'
        if os.path.exists(file):
            df_history = pd.read_excel(file)
            for _, row in df_history.iterrows():
                self.shot_table.insert('', 'end', values=(
                    row['Shot No'],
                    round(row['Velocity (m/s)'], 2),
                    round(row['Predicted Velocity'], 2),
                    f"{round(row['Error %'], 2)}%"
                ))
            self.shot_no = len(df_history) + 1

    def read_arduino(self):

        while True:
            try:
                if self.ser.in_waiting > 0:

                    line = self.ser.readline().decode().strip()

                    print("Arduino:", line)

                    if line.startswith("TIME="):

                        time_us = float(line.replace("TIME=",""))

                        self.root.after(0, self.calculate_velocity_from_time, time_us)

            except Exception as e:
                print("Serial read error:", e)

    def update_actual_velocity_value(self, velocity):
        self.actual_velocity = velocity
        self.actual_label.config(text=f'{velocity:.2f} m/s')
        self.calculate_error()
    def calculate_velocity_from_time(self, time_us):

        try:

            distance = float(self.sensor_distance.get())

            time_seconds = time_us / 1000000

            velocity = distance / time_seconds

            self.actual_velocity = velocity

            self.actual_label.config(text=f"{velocity:.2f} m/s")

            self.calculate_error()

        except:
            pass
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — POWDER MASS
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab2(self, tab):
        outer, card = self.create_card(tab, 'POWDER MASS ESTIMATION', self.C['warn'])
        outer.pack(fill='both', expand=True, padx=8, pady=8)

        ct = tk.Frame(card, bg=self.C['surface'])
        ct.pack(fill='both', expand=True, padx=16, pady=14)

        r = 0
        self.calibre2        = self.create_input_field(ct, 'Calibre',               r, 0, True, calibres);         r+=1
        self.projectile_type2= self.create_input_field(ct, 'Projectile Type',        r, 0, True, projectile_types); r+=1
        self.dimension2      = self.create_input_field(ct, 'Projectile Dimension',   r, 0);                         r+=1
        self.mass2           = self.create_input_field(ct, 'Projectile Mass',        r, 0);                         r+=1
        self.total_mass2     = self.create_input_field(ct, 'Total Mass w/ Sabot',    r, 0);                         r+=1
        self.pressure2       = self.create_input_field(ct, 'Petal Burst Pressure',   r, 0);                         r+=1

        self.shape2   = self.create_input_field(ct, 'Shape',            0, 1, True, shapes)
        self.breadth2 = self.create_input_field(ct, 'Breadth',          1, 1)
        self.height2  = self.create_input_field(ct, 'Height',           2, 1)
        self.material2= self.create_input_field(ct, 'Material',         3, 1, True, materials)
        self.cdrag2   = self.create_input_field(ct, 'Drag Coefficient', 4, 1)
        self.sa2      = self.create_input_field(ct, 'Surface Area',     5, 1)
        self.vol2     = self.create_input_field(ct, 'Volume',           6, 1)
        self.sabot2   = self.create_input_field(ct, 'Sabot Length',     7, 1)
        self.target_velocity = self.create_input_field(ct, 'Target Velocity', 8, 0)

        self._sep(ct, 9, 4)

        btn_f = tk.Frame(ct, bg=self.C['surface'])
        btn_f.grid(row=10, column=0, columnspan=4, pady=14)
        self.create_button(btn_f, 'ESTIMATE POWDER MASS',
                           self.estimate_powder, self.C['warn']).pack()

        # Result display
        res_blk = tk.Frame(ct, bg=self.C['surface2'],
                           highlightbackground=self.C['warn'],
                           highlightthickness=1)
        res_blk.grid(row=11, column=0, columnspan=4, pady=14, padx=10, sticky='ew')
        tk.Label(res_blk, text='ESTIMATED POWDER MASS',
                 bg=self.C['surface2'], fg=self.C['text_dim'],
                 font=('Courier New', 10)).pack(pady=(8, 0))
        self.powder_result = tk.Label(res_blk, text='--- g',
                                      bg=self.C['surface2'],
                                      fg=self.C['warn'],
                                      font=('Courier New', 24, 'bold'))
        self.powder_result.pack(pady=(0, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — PHYSICS
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab3(self, tab):
        outer, card = self.create_card(tab, 'PHYSICS CALCULATIONS', self.C['accent'])
        outer.pack(fill='both', expand=True, padx=8, pady=8)

        ct = tk.Frame(card, bg=self.C['surface'])
        ct.pack(fill='both', expand=True, padx=20, pady=20)

        params = [
            ('⚡  KINETIC ENERGY',  'ke',       'Joules (J)',  self.C['accent']),
            ('📈  MOMENTUM',        'momentum', 'kg·m/s',      self.C['accent2']),
            ('🔧  FORCE',           'force',    'Newtons (N)', self.C['warn']),
            ('🚀  ACCELERATION',    'acc',      'm/s²',        self.C['danger']),
        ]
        self.physics_labels = {}
        for i, (title, attr, unit, fg) in enumerate(params):
            r, c = divmod(i, 2)
            blk = tk.Frame(ct, bg=self.C['surface2'],
                           highlightbackground=self.C['border'],
                           highlightthickness=1)
            blk.grid(row=r, column=c, padx=12, pady=12, sticky='nsew', ipady=20, ipadx=20)

            tk.Frame(blk, bg=fg, height=3).pack(fill='x')
            tk.Label(blk, text=title, bg=self.C['surface2'],
                     fg=fg, font=('Courier New', 13, 'bold')).pack(pady=(14, 4))
            val_lbl = tk.Label(blk, text='--', bg=self.C['surface2'],
                               fg=self.C['text'], font=('Courier New', 32, 'bold'))
            val_lbl.pack(pady=(0, 4))
            tk.Label(blk, text=unit, bg=self.C['surface2'],
                     fg=self.C['text_dim'], font=('Courier New', 11)).pack(pady=(0, 14))
            self.physics_labels[attr] = val_lbl

        for i in range(2):
            ct.grid_columnconfigure(i, weight=1)
            ct.grid_rowconfigure(i, weight=1)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — GRAPHS
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab4(self, tab):
        outer, card = self.create_card(tab, 'GRAPHICAL ANALYSIS', self.C['accent'])
        outer.pack(fill='both', expand=True, padx=8, pady=8)

        bg_hex = self.C['surface']
        plot_bg = self.C['bg']

        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(2, 3, figsize=(13, 6))
        self.fig.patch.set_facecolor(bg_hex)

        for ax in self.axs.flat:
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=self.C['text_dim'], labelsize=7)
            ax.xaxis.label.set_color(self.C['text_dim'])
            ax.yaxis.label.set_color(self.C['text_dim'])
            ax.title.set_color(self.C['accent'])
            for spine in ax.spines.values():
                spine.set_color(self.C['border'])
            ax.grid(True, alpha=0.15, color=self.C['border'])

        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, card)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)
        self.canvas.get_tk_widget().configure(bg=bg_hex)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — MODEL ACCURACY
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab5(self, tab):
        outer, card = self.create_card(tab, 'MODEL ACCURACY', self.C['accent2'])
        outer.pack(fill='both', expand=True, padx=8, pady=8)

        left_f  = tk.Frame(card, bg=self.C['surface'])
        left_f.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        right_f = tk.Frame(card, bg=self.C['surface'])
        right_f.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        cols = ('Model', 'Accuracy')
        tree = ttk.Treeview(left_f, columns=cols, show='headings',
                            height=12, style='T.Treeview')
        tree.heading('Model',    text='MODEL')
        tree.heading('Accuracy', text='ACCURACY')
        tree.column('Model',    width=220)
        tree.column('Accuracy', width=120, anchor='center')
        tree.pack(side='left', fill='both', expand=True)

        sb = ttk.Scrollbar(left_f, orient='vertical', command=tree.yview,
                           style='T.Vertical.TScrollbar')
        sb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=sb.set)

        try:
            results = pd.read_csv('models/model_results.csv')
            for _, row in results.iterrows():
                tree.insert('', 'end', values=(row['Model'], f"{row['Accuracy']:.2f}%"))

            fig2, ax = plt.subplots(figsize=(5, 4))
            fig2.patch.set_facecolor(self.C['surface'])
            ax.set_facecolor(self.C['bg'])

            bar_colors = [self.C['accent'], self.C['accent2'],
                          self.C['warn'], self.C['danger']]
            bars = ax.bar(results['Model'], results['Accuracy'],
                          color=[bar_colors[i % len(bar_colors)]
                                 for i in range(len(results))],
                          edgecolor=self.C['border'], linewidth=0.8)

            ax.set_title('Model Performance', color=self.C['accent'],
                         fontsize=11, fontweight='bold', fontfamily='Courier New')
            ax.set_xlabel('Model',    color=self.C['text_dim'],
                          fontfamily='Courier New', fontsize=8)
            ax.set_ylabel('Accuracy (%)', color=self.C['text_dim'],
                          fontfamily='Courier New', fontsize=8)
            ax.tick_params(colors=self.C['text_dim'], labelsize=7)
            ax.set_ylim(0, max(results['Accuracy']) * 1.15)
            for spine in ax.spines.values():
                spine.set_color(self.C['border'])
            ax.grid(True, axis='y', alpha=0.15, color=self.C['border'])

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom',
                        color=self.C['text'], fontsize=8,
                        fontfamily='Courier New')

            fig2.tight_layout()
            c2 = FigureCanvasTkAgg(fig2, right_f)
            c2.get_tk_widget().pack(fill='both', expand=True)
            c2.get_tk_widget().configure(bg=self.C['surface'])

        except:
            tree.insert('', 'end', values=('No data available', ''))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — DATASET
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab6(self, tab):
        outer, card = self.create_card(tab, 'DATASET PREVIEW', self.C['accent'])
        outer.pack(fill='both', expand=True, padx=8, pady=8)

        tree_f = tk.Frame(card, bg=self.C['surface'])
        tree_f.pack(fill='both', expand=True, padx=8, pady=8)

        tree = ttk.Treeview(tree_f, style='T.Treeview')
        tree['columns'] = list(df.columns)
        tree['show'] = 'headings'
        for col in df.columns:
            tree.heading(col, text=col.upper())
            tree.column(col, width=110, anchor='center')
        for _, row in df.head(50).iterrows():
            tree.insert('', 'end', values=list(row))

        vsb = ttk.Scrollbar(tree_f, orient='vertical',   command=tree.yview,
                            style='T.Vertical.TScrollbar')
        hsb = ttk.Scrollbar(tree_f, orient='horizontal',  command=tree.xview,
                            style='T.Horizontal.TScrollbar')
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_f.grid_rowconfigure(0, weight=1)
        tree_f.grid_columnconfigure(0, weight=1)

        # Status bar
        sb_f = tk.Frame(card, bg=self.C['surface2'])
        sb_f.pack(fill='x', padx=8, pady=(0, 8))
        tk.Label(sb_f,
                 text=f'  ◈  {len(df)} RECORDS  ·  {len(df.columns)} FIELDS  ·  SHOWING FIRST 50',
                 bg=self.C['surface2'], fg=self.C['text_dim'],
                 font=self.font_status).pack(side='left', pady=5)
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — REPORT
    # ══════════════════════════════════════════════════════════════════════════
    def build_tab7(self, tab):

        outer, card = self.create_card(tab, "PDF REPORT GENERATOR", self.C['accent'])
        outer.pack(fill='both', expand=True, padx=8, pady=8)

        frame = tk.Frame(card, bg=self.C['surface'])
        frame.pack(expand=True)

        tk.Label(
            frame,
            text="BALLISTIC TEST REPORT",
            bg=self.C['surface'],
            fg=self.C['accent'],
            font=('Courier New',20,'bold')
        ).pack(pady=20)

        btn = tk.Button(
            frame,
            text="DOWNLOAD PDF REPORT",
            command=self.generate_pdf,
            font=('Courier New',14,'bold'),
            bg=self.C['surface2'],
            fg=self.C['accent'],
            padx=20,
            pady=10
        )
        btn.pack(pady=30)
    def generate_pdf(self):
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import HexColor

        file_name = "ballistics_report.pdf"
        c = canvas.Canvas(file_name, pagesize=A4)

        width, height = A4
        y = height - 50

        # ===== TITLE =====
        c.setFont("Helvetica-Bold",20)
        c.setFillColor(HexColor("#1c2030"))
        c.drawCentredString(width/2, y, "BALLISTICS TEST REPORT")

        y -= 30
        c.setFont("Helvetica",10)
        c.drawCentredString(width/2, y, f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y -= 40

        # ================= INPUT PARAMETERS =================

        c.setFont("Helvetica-Bold",14)
        c.drawString(40,y,"INPUT PARAMETERS")
        y -= 20

        c.setFont("Courier",10)

        inputs = [
            ("Calibre", self.calibre.get()),
            ("Projectile Type", self.projectile_type.get()),
            ("Projectile Dimension", self.dimension.get()),
            ("Projectile Mass", self.mass.get()),
            ("Total Mass with Sabot", self.total_mass.get()),
            ("Powder Mass", self.powder.get()),
            ("Petal Burst Pressure", self.pressure.get()),
            ("Sensor Distance", self.sensor_distance.get()),

            ("Shape", self.shape.get()),
            ("Breadth", self.breadth.get()),
            ("Height", self.height.get()),
            ("Material", self.material.get()),
            ("Drag Coefficient", self.cdrag.get()),
            ("Surface Area", self.sa.get()),
            ("Volume", self.vol.get()),
            ("Sabot Length", self.sabot.get())
        ]

        for label,value in inputs:
            c.drawString(50,y,f"{label:<25} : {value}")
            y -= 15

        y -= 10

        # ================= RESULTS =================

        c.setFont("Helvetica-Bold",14)
        c.drawString(40,y,"RESULTS")
        y -= 20

        c.setFont("Courier",11)

        c.drawString(50,y,f"Predicted Velocity  : {self.predicted_velocity:.2f} m/s")
        y -= 20

        c.drawString(50,y,f"Actual Velocity     : {self.actual_velocity:.2f} m/s")
        y -= 20

        if self.predicted_velocity != 0:
            error = abs(self.predicted_velocity-self.actual_velocity)/self.predicted_velocity*100
        else:
            error = 0

        c.drawString(50,y,f"Error Percentage    : {error:.2f} %")

        y -= 40

        # ================= PHYSICS =================

        c.setFont("Helvetica-Bold",14)
        c.drawString(40,y,"PHYSICS CALCULATIONS")

        y -= 20
        c.setFont("Courier",11)

        c.drawString(50,y,f"Kinetic Energy : {self.physics_labels['ke'].cget('text')} J")
        y -= 20

        c.drawString(50,y,f"Momentum       : {self.physics_labels['momentum'].cget('text')} kg·m/s")
        y -= 20

        c.drawString(50,y,f"Force          : {self.physics_labels['force'].cget('text')} N")
        y -= 20

        c.drawString(50,y,f"Acceleration   : {self.physics_labels['acc'].cget('text')} m/s²")
        y -= 40
        # ================= SHOT HISTORY =================

        c.setFont("Helvetica-Bold",14)
        c.drawString(40,y,"SHOT HISTORY (SCREEN DATA)")
        y -= 20

        table_data = [["Shot","Time","Velocity","Predicted","Error"]]

        for row in self.shot_table.get_children():

            values = self.shot_table.item(row)['values']

            table_data.append(values)

        table = Table(table_data)

        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.grey),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("GRID",(0,0),(-1,-1),1,colors.black),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ]))

        table.wrapOn(c,width,height)

        table_height = len(table_data)*20

        table.drawOn(c,40,y-table_height)

        y -= (table_height+20)
        # ================= GRAPHS =================

        self.fig.savefig("graphs.png")

        c.showPage()

        c.setFont("Helvetica-Bold",16)
        c.drawCentredString(width/2,height-50,"BALLISTIC ANALYSIS GRAPHS")

        c.drawImage("graphs.png",50,200,width=500,height=350)

        c.save()

        print("PDF report saved:",file_name)

        import webbrowser
        webbrowser.open(file_name)
    # ══════════════════════════════════════════════════════════════════════════
    # LOGIC — unchanged from original
    # ══════════════════════════════════════════════════════════════════════════
    def predict(self):
        try:
            inputs = [
                float(self.calibre.get().replace(' mm', '')),
                float(self.dimension.get()),
                float(self.mass.get()),
                float(self.powder.get()),
                float(self.sa.get()),
                float(self.vol.get()),
                float(self.cdrag.get())
            ]
            velocity = predict_velocity(inputs)
            self.predicted_velocity = velocity
            self.velocity = velocity
            self.predicted_label.config(text=f'{velocity:.2f} m/s')
            self.update_physics()
            self.update_graphs()
        except:
            self.predicted_label.config(text='INVALID INPUT')

    def estimate_powder(self):
        try:
            mass     = float(self.mass2.get())
            velocity = float(self.target_velocity.get())
            powder   = (mass * velocity) / 1000
            self.powder_result.config(text=f'{powder:.2f} g')
        except:
            self.powder_result.config(text='INVALID INPUT')

    def update_physics(self):
        try:
            m = float(self.mass.get())
            v = self.velocity
            ke       = 0.5 * m * v ** 2
            momentum = m * v
            a        = v ** 2 / 2
            force    = m * a
            self.physics_labels['ke'].config(text=f'{ke:.2f}')
            self.physics_labels['momentum'].config(text=f'{momentum:.2f}')
            self.physics_labels['force'].config(text=f'{force:.2f}')
            self.physics_labels['acc'].config(text=f'{a:.2f}')
        except:
            pass

    def update_graphs(self):
        try:
            mass     = float(self.mass.get()) / 1000
            velocity = self.velocity
            drag     = float(self.cdrag.get())
            area     = float(self.sa.get()) / 10000
            rho      = 1.225
            time     = np.linspace(0, 0.02, 200)
            pressure_peak  = float(self.pressure.get())
            pressure       = pressure_peak * np.exp(-time * 120)
            velocity_curve = velocity * (1 - np.exp(-time * 80))
            distance       = np.cumsum(velocity_curve) * (time[1] - time[0])
            energy         = 0.5 * mass * velocity_curve ** 2
            momentum       = mass * velocity_curve
            drag_force     = 0.5 * rho * velocity_curve ** 2 * drag * area

            plot_colors = [
                self.C['accent'], self.C['accent2'],
                self.C['warn'],   self.C['danger'],
                '#b388ff',        self.C['text_mid']
            ]
            plots = [
                (self.axs[0, 0], time * 1000, pressure,       'Pressure vs Time',      'Time (ms)',    'Pressure'),
                (self.axs[0, 1], time * 1000, velocity_curve, 'Velocity vs Time',      'Time (ms)',    'Velocity (m/s)'),
                (self.axs[0, 2], time * 1000, distance,       'Distance vs Time',      'Time (ms)',    'Distance (m)'),
                (self.axs[1, 0], time * 1000, energy,         'Energy vs Time',         'Time (ms)',    'Energy (J)'),
                (self.axs[1, 1], time * 1000, momentum,       'Momentum vs Time',       'Time (ms)',    'Momentum'),
                (self.axs[1, 2], velocity_curve, drag_force,  'Drag Force vs Velocity', 'Velocity (m/s)', 'Drag Force'),
            ]

            for i, (ax, x, y, title, xl, yl) in enumerate(plots):
                ax.clear()
                ax.set_facecolor(self.C['bg'])
                ax.plot(x, y, color=plot_colors[i], linewidth=2.2)
                ax.fill_between(x, y, alpha=0.08, color=plot_colors[i])
                ax.set_title(title, color=self.C['accent'], fontsize=9,
                             fontweight='bold', fontfamily='Courier New')
                ax.set_xlabel(xl, color=self.C['text_dim'], fontsize=7,
                              fontfamily='Courier New')
                ax.set_ylabel(yl, color=self.C['text_dim'], fontsize=7,
                              fontfamily='Courier New')
                ax.tick_params(colors=self.C['text_dim'], labelsize=6)
                ax.grid(True, alpha=0.12, color=self.C['border'])
                for spine in ax.spines.values():
                    spine.set_color(self.C['border'])

            self.fig.tight_layout(pad=2.0)
            self.canvas.draw()
        except:
            pass


if __name__ == '__main__':
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()