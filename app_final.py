"""
Peng-Robinson EOS — Equilibrio de Fases
v12: Estructura multi-widget. Títulos como QLabel, datos como QTableWidget pequeños.
     Esto elimina el conflicto de QSS y permite colorear celda por celda.
"""
import sys, os, copy
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QDoubleSpinBox, QGridLayout, QFrame, QHeaderView,
    QCheckBox, QMessageBox, QStatusBar, QAbstractItemView, QScrollArea
)
import matplotlib
matplotlib.use('QtAgg')
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QIcon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine3 import (
    COMPONENTES, NOMBRES, PM, TC, PC, OMEGA, KIJ_DEFAULT, NC,
    calcular, R_GAS
)
from tab_envelope import TabEnvolvente

kij_user = copy.deepcopy(KIJ_DEFAULT)

# ── Paleta ────────────────────────────────────────────────────
WHITE    = "#FFFFFF"
GRAY_TIT = "#A8A8A8"   # plomo oscuro para títulos / cabeceras
GRAY_LBL = "#D0D0D0"   # plomo medio para etiquetas
GRAY_RES = "#E8E8E8"   # plomo claro para celdas de resultado (vacías)
BORDER   = "#888888"
TEXT     = "#000000"
TEXT_DIM = "#555555"
TEXT_RES = "#000080"   # azul oscuro para resultados
FONT_F   = "Arial Narrow"
FS       = 10

# ── Helpers de color ─────────────────────────────────────────
def _brush(hex_color):
    return QBrush(QColor(hex_color), Qt.BrushStyle.SolidPattern)

def cell(text, bg=WHITE, color=TEXT,
         align=Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,
         editable=False):
    """Crea un QTableWidgetItem con color de fondo explícito."""
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(align)
    it.setBackground(_brush(bg))
    it.setForeground(_brush(color))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it

def title_label(text):
    """Barra de título oscura."""
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setFixedHeight(22)
    lbl.setStyleSheet(
        f'background:{GRAY_TIT}; color:{TEXT}; '
        f'font-family:"{FONT_F}"; font-size:{FS}pt; '
        f'padding:0px 6px; border:1px solid {BORDER};'
    )
    return lbl

def section_label(text, left=False):
    """Barra de sección (plomo medio)."""
    lbl = QLabel(text)
    align = (Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
             if left else Qt.AlignmentFlag.AlignCenter)
    lbl.setAlignment(align)
    lbl.setFixedHeight(22)
    lbl.setStyleSheet(
        f'background:{GRAY_LBL}; color:{TEXT}; '
        f'font-family:"{FONT_F}"; font-size:{FS}pt; '
        f'padding:0px 8px; border:1px solid {BORDER};'
    )
    return lbl

def make_table(rows, cols, row_h=22):
    """Tabla sin cabeceras, sin scroll, tamaño fijo."""
    t = QTableWidget(rows, cols)
    t.horizontalHeader().hide()
    t.verticalHeader().hide()
    t.setShowGrid(True)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setStyleSheet(
        f'QTableWidget {{ border:1px solid {BORDER}; '
        f'font-family:"{FONT_F}"; font-size:{FS}pt; gridline-color:{BORDER}; }}'
        f'QTableWidget::item {{ padding:2px 6px; }}'
    )
    for r in range(rows):
        t.setRowHeight(r, row_h)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return t

def fix_table_size(t):
    """Ajusta el tamaño de la tabla a su contenido."""
    w = sum(t.columnWidth(c) for c in range(t.columnCount())) + 2
    h = sum(t.rowHeight(r) for r in range(t.rowCount())) + 2
    t.setFixedSize(w, h)

# ── Dimensiones ──────────────────────────────────────────────
W_LBL  = 255   # columna de etiqueta
W_VAL  = 140   # columna de valor (Vapor o Líquido)
W_COMP = 290   # columna nombre de componente
ROW_H  = 22

# ── Worker ────────────────────────────────────────────────────
class Worker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, z, T, P, kij):
        super().__init__()
        self.z=z; self.T=T; self.P=P; self.kij=kij
    def run(self):
        try:    self.done.emit(calcular(self.z, self.T, self.P, self.kij))
        except Exception as e: self.error.emit(str(e))

# ══════════════════════════════════════════════════════════════
# Tab 1 — Equilibrio de Fases
# ══════════════════════════════════════════════════════════════
class TabEquilibrio(QWidget):
    def __init__(self):
        super().__init__()
        self.worker      = None
        self.last_result = None
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        outer.setSpacing(0)

        # Centrar todo al ancho de la tabla
        hc = QHBoxLayout()
        hc.setContentsMargins(0,0,0,0)
        hc.addStretch()

        # Ancho total = W_COMP + 3*W_VAL
        TW = W_COMP + 3*W_VAL   # 290 + 420 = 710

        box = QWidget()
        box.setFixedWidth(TW)
        root = QVBoxLayout(box)
        root.setContentsMargins(0, 8, 0, 8)
        root.setSpacing(4)

        # ── Fila entradas P/T + checkbox + botón ─────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        pin = QFrame()
        pin.setStyleSheet(f'border:1px solid {BORDER};')
        gl = QGridLayout(pin)
        gl.setContentsMargins(6,4,6,4); gl.setSpacing(4)

        def inp_lbl(txt):
            l = QLabel(txt)
            l.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            l.setStyleSheet(
                f'background:{GRAY_LBL};border:1px solid {BORDER};'
                f'padding:2px 6px;font-family:"{FONT_F}";font-size:{FS}pt;')
            l.setFixedHeight(22)
            return l

        gl.addWidget(inp_lbl("Presion (psi):"), 0, 0)
        self.sp_P = QDoubleSpinBox()
        self.sp_P.setRange(1,15000); self.sp_P.setValue(200); self.sp_P.setDecimals(2)
        self.sp_P.setFixedHeight(22); self.sp_P.setFixedWidth(110)
        self.sp_P.setStyleSheet(
            f'QDoubleSpinBox {{ background:{WHITE};border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt; }}')
        gl.addWidget(self.sp_P, 0, 1)

        gl.addWidget(inp_lbl("Temperatura (°R):"), 1, 0)
        self.sp_T = QDoubleSpinBox()
        self.sp_T.setRange(1.0, 9999.99)
        self.sp_T.setValue(350.0)
        self.sp_T.setDecimals(2)
        self.sp_T.setFixedHeight(22); self.sp_T.setFixedWidth(110)
        self.sp_T.setStyleSheet(
            f'QDoubleSpinBox {{ background:{WHITE};border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt; }}')
        self.sp_T.valueChanged.connect(
            lambda v: self.lbl_F.setText(f"({v-459.67:.1f} °F)"))
        gl.addWidget(self.sp_T, 1, 1)

        self.lbl_F = QLabel("(-109.7 °F)")
        self.lbl_F.setStyleSheet(
            f'color:{TEXT_DIM};font-size:9pt;background:transparent;'
            f'font-family:"{FONT_F}";')
        self.lbl_F.setAlignment(
            Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        gl.addWidget(self.lbl_F, 2, 1)

        top.addWidget(pin,
            alignment=Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        top.addStretch()

        rp = QHBoxLayout(); rp.setSpacing(8)

        self.chk = QCheckBox("Fraccion masica")
        self.chk.setStyleSheet(
            f'QCheckBox {{ font-family:"{FONT_F}";font-size:{FS}pt;'
            f'background:transparent; spacing:6px; }}')
        self.chk.stateChanged.connect(self._on_chk)
        rp.addWidget(self.chk, alignment=Qt.AlignmentFlag.AlignVCenter)

        btn_n = QPushButton("Normalizar")
        btn_n.setFixedWidth(100)
        btn_n.setStyleSheet(
            f'background:{GRAY_LBL};border:2px outset {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;min-height:22px;')
        btn_n.clicked.connect(self.normalizar)
        rp.addWidget(btn_n, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.btn = QPushButton("Realizar Calculo")
        self.btn.setFixedWidth(130)
        self.btn.setStyleSheet(
            f'background:{GRAY_LBL};border:2px outset {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;min-height:22px;')
        self.btn.clicked.connect(self.calcular)
        rp.addWidget(self.btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        top.addLayout(rp)
        root.addLayout(top)

        # ── TÍTULO principal ──────────────────────────────────
        root.addWidget(title_label("ThermoPhase — Equilibrio de Fases"))

        # ── BLOQUE RESUMEN ────────────────────────────────────
        # Título de sección
        root.addWidget(section_label("Resumen de los calculos:", left=True))

        # Cabecera de columnas del resumen (plomo medio)
        # Anchos del resumen = mismos que composicion para alinear columnas
        WR0 = W_COMP        # 290 — etiqueta
        WR1 = 100           # Mezcla
        WR2 = W_VAL         # Vapor  (140)
        WR3 = W_VAL         # Liquida(140)
        # Total = 290+100+140+140 = 670 — coincide con W_COMP+3*W_VAL si W_VAL=126.6
        # Ajustamos W_VAL para que todo sume igual:
        # W_COMP + 3*W_VAL = WR0+WR1+WR2+WR3 → 290+3*W_VAL = 290+100+2*W_VAL → W_VAL=100
        # Mejor: fijamos total = W_COMP+W_VAL*3 y distribuimos:
        # WR0=W_COMP, WR1+WR2+WR3 = W_VAL*3 → WR1=W_VAL-40, WR2=WR3=(W_VAL*3-(W_VAL-40))/2
        WR1 = W_VAL   # misma anchura que Vapor y Liquida
        WR2 = W_VAL
        WR3 = W_VAL
        hdr_res = make_table(1, 4)
        hdr_res.setColumnWidth(0, W_COMP)
        hdr_res.setColumnWidth(1, WR1)
        hdr_res.setColumnWidth(2, WR2)
        hdr_res.setColumnWidth(3, WR3)
        hdr_res.setItem(0,0, cell("", bg=GRAY_LBL))
        hdr_res.setItem(0,1, cell("Mezcla", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))
        hdr_res.setItem(0,2, cell("Fase Vapor", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))
        hdr_res.setItem(0,3, cell("Fase Liquida", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))
        fix_table_size(hdr_res)
        root.addWidget(hdr_res)

        # Tabla de datos del resumen
        # Filas: ff_mol, ff_mas, grav, dens, Z, PM
        # Tabla resumen: 4 columnas
        # col0=etiqueta, col1=mezcla(solo dens y PM), col2=Fase Vapor, col3=Fase Liquida
        # Para filas sin mezcla: col1 queda plomo/vacía, col2 y col3 tienen los valores
        self.tbl_res = make_table(6, 4)
        self.tbl_res.setColumnWidth(0, W_COMP)
        self.tbl_res.setColumnWidth(1, W_VAL)
        self.tbl_res.setColumnWidth(2, W_VAL)
        self.tbl_res.setColumnWidth(3, W_VAL)

        res_labels = [
            "Fase fraccion [molar]:",
            "Fase fraccion [masica]:",
            "Gravedad especifica:",
            "Densidad masica [lb/ft3]:",
            "Factor de compresibilidad:",
            "Peso molecular:",
        ]
        self.res_has_mix = {3, 5}   # Densidad y PM tienen valor de mezcla

        for i, lbl_txt in enumerate(res_labels):
            self.tbl_res.setItem(i, 0, cell(lbl_txt, bg=GRAY_LBL))
            # Celdas de resultado: fondo plomo claro (GRAY_RES)
            self.tbl_res.setItem(i, 1, cell("", bg=GRAY_RES))
            self.tbl_res.setItem(i, 2, cell("", bg=GRAY_RES))
            self.tbl_res.setItem(i, 3, cell("", bg=GRAY_RES))

        fix_table_size(self.tbl_res)
        root.addWidget(self.tbl_res)

        # ── BLOQUE COMPOSICIÓN ────────────────────────────────
        root.addWidget(section_label("Composicion de las fases:", left=True))

        # Cabecera de composición (2 niveles)
        hdr_comp = make_table(2, 4)
        hdr_comp.setRowHeight(0, ROW_H)
        hdr_comp.setRowHeight(1, ROW_H)
        hdr_comp.setColumnWidth(0, W_COMP)
        for c in [1,2,3]: hdr_comp.setColumnWidth(c, W_VAL)

        hdr_comp.setItem(0,0, cell("Componente", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))
        hdr_comp.setItem(0,1, cell("Composicion General", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))
        hdr_comp.setItem(0,2, cell("Fase Vapor", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))
        hdr_comp.setItem(0,3, cell("Fase Liquida", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter))

        hdr_comp.setItem(1,0, cell("", bg=GRAY_LBL))
        self.hdr_comp_gen  = cell("Fraccion Molar", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter)
        self.hdr_comp_vap  = cell("Fraccion molar", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter)
        self.hdr_comp_liq  = cell("Fraccion molar", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignCenter)
        hdr_comp.setItem(1,1, self.hdr_comp_gen)
        hdr_comp.setItem(1,2, self.hdr_comp_vap)
        hdr_comp.setItem(1,3, self.hdr_comp_liq)
        fix_table_size(hdr_comp)
        root.addWidget(hdr_comp)

        # Tabla de componentes
        self.tbl_comp = make_table(NC+1, 4)
        self.tbl_comp.setColumnWidth(0, W_COMP)
        for c in [1,2,3]: self.tbl_comp.setColumnWidth(c, W_VAL)

        z_def = [0,0,0.9,0.05,0.05,0,0,0,0,0,0,0,0]
        for i in range(NC):
            self.tbl_comp.setItem(i, 0, cell(
                NOMBRES[i], bg=GRAY_LBL,
                align=Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter))
            self.tbl_comp.setItem(i, 1, cell(
                f"{z_def[i]:.4f}", bg=WHITE, editable=True))
            self.tbl_comp.setItem(i, 2, cell("", bg=GRAY_RES, color=TEXT_RES))
            self.tbl_comp.setItem(i, 3, cell("", bg=GRAY_RES, color=TEXT_RES))

        # Fila de sumatorias dentro de tbl_comp (fila NC)
        self.tbl_comp.setItem(NC, 0, cell("Sumatorias:", bg=GRAY_LBL,
            align=Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter))
        self.tbl_comp.setItem(NC, 1, cell("1.0000", bg=WHITE))
        self.tbl_comp.setItem(NC, 2, cell("", bg=GRAY_RES))
        self.tbl_comp.setItem(NC, 3, cell("", bg=GRAY_RES))
        self.sum_row = NC  # índice de la fila sumatoria dentro de tbl_comp
        fix_table_size(self.tbl_comp)
        self.tbl_comp.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.tbl_comp)



        hc.addWidget(box)
        hc.addStretch()
        outer.addLayout(hc)

    # ── Helpers de entrada ───────────────────────────────────
    def get_T(self):
        """Lee la temperatura del QDoubleSpinBox."""
        return self.sp_T.value()

    def get_P(self):
        """Lee la presión del QLineEdit de forma segura."""
        try:
            val = float(self.sp_P.text().replace(',', '.'))
            return val if val > 0 else 200.0
        except ValueError:
            return 200.0

    # ── Handlers ─────────────────────────────────────────────
    def _on_chk(self):
        masa = self.chk.isChecked()
        # Col 1 (Composicion General) SIEMPRE "Fraccion Molar"
        self.hdr_comp_vap.setText(
            "Fraccion masica" if masa else "Fraccion molar")
        self.hdr_comp_liq.setText(
            "Fraccion masica" if masa else "Fraccion molar")
        if self.last_result:
            self._render(self.last_result)

    def _on_item_changed(self, item):
        if item.column() != 1: return
        if item.row() == self.sum_row: return  # no procesar fila sumatorias
        self._upd_suma()

    def get_z(self):
        z = []
        for i in range(NC):
            try: z.append(float(self.tbl_comp.item(i,1).text()))
            except: z.append(0.0)
        return z

    def _upd_suma(self):
        s = sum(self.get_z())
        self.tbl_comp.blockSignals(True)
        self.tbl_comp.item(self.sum_row,1).setText(f"{s:.4f}")
        self.tbl_comp.blockSignals(False)

    def normalizar(self):
        z = self.get_z(); s = sum(z)
        if s <= 0: return
        self.tbl_comp.blockSignals(True)
        for i in range(NC):
            self.tbl_comp.item(i,1).setText(f"{z[i]/s:.4f}")
        self.tbl_comp.blockSignals(False)
        self._upd_suma()  # actualiza fila sumatorias

    def calcular(self):
        z = self.get_z()
        if abs(sum(z)-1.0) > 1e-3:
            QMessageBox.warning(self, "Composicion",
                "La suma de fracciones debe ser 1.0")
            return
        self.btn.setEnabled(False); self.btn.setText("Calculando...")
        self.worker = Worker(z, self.get_T(), self.get_P(), kij_user)
        self.worker.done.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_error(self, msg):
        self.btn.setEnabled(True); self.btn.setText("Realizar Calculo")
        QMessageBox.critical(self, "Error", msg)

    def _on_result(self, r):
        self.btn.setEnabled(True); self.btn.setText("Realizar Calculo")
        self.last_result = r
        self._render(r)

    def _render(self, r):
        masa = self.chk.isChecked()
        V=r["V"]; L=r["L"]
        ZV=r.get("ZV"); ZL=r.get("ZL")
        x=r["x"]; y=r["y"]
        PM_v=r.get("PM_v"); PM_l=r.get("PM_l"); PM_z=r.get("PM_z")
        rho_v=r.get("rho_v"); rho_l=r.get("rho_l")
        sg_v=r.get("sg_v"); sg_l=r.get("sg_l")
        Vm=r["Vm"]; Lm=r["Lm"]
        modo=r["modo"]

        rho_z = None
        if rho_v and rho_l:
            inv = (Vm/rho_v if rho_v>0 else 0)+(Lm/rho_l if rho_l>0 else 0)
            if inv>0: rho_z = 1.0/inv
        elif rho_l: rho_z = rho_l
        elif rho_v: rho_z = rho_v

        def f(v, d=4):
            return f"{v:.{d}f}" if v is not None else ""

        def paint(item, txt):
            item.setText(txt)
            if txt:
                item.setBackground(_brush(WHITE))
                item.setForeground(_brush(TEXT_RES))
            else:
                item.setBackground(_brush(GRAY_RES))
                item.setForeground(_brush(TEXT))

        # Resumen — 6 filas × 3 cols (etiqueta, vapor, liquida)
        # col 0 = etiqueta (plomo siempre)
        # col 1 = Fase Vapor  → blanco si tiene valor
        # col 2 = Fase Liquida→ blanco si tiene valor
        #
        # Filas dens (3) y PM (5) también usan col0 para valor de mezcla
        data = [None]*6
        if modo == "liquido_unico":
            data[0] = ("", f(V) if V>0 else "", f(L) if L>0 else "")
            data[1] = ("", f(Vm) if V>0 else "", f(Lm) if L>0 else "")
            data[2] = ("", "", f(sg_l))
            data[3] = (f(rho_z), "", f(rho_l))
            data[4] = ("", "", f(ZL))
            data[5] = (f(PM_z), "", f(PM_l))
        elif modo == "vapor_unico":
            data[0] = ("", f(V) if V>0 else "", "")
            data[1] = ("", f(Vm) if V>0 else "", "")
            data[2] = ("", f(sg_v), "")
            data[3] = (f(rho_z), f(rho_v), "")
            data[4] = ("", f(ZV), "")
            data[5] = (f(PM_z), f(PM_v), "")
        else:
            data[0] = ("", f(V), f(L))
            data[1] = ("", f(Vm), f(Lm))
            data[2] = ("", f(sg_v), f(sg_l))
            data[3] = (f(rho_z), f(rho_v), f(rho_l))
            data[4] = ("", f(ZV), f(ZL))
            data[5] = (f(PM_z), f(PM_v), f(PM_l))

        for i, (mix, vap, liq) in enumerate(data):
            # col0=etiqueta (plomo fijo), col1=mezcla, col2=vapor, col3=liquida
            if i in self.res_has_mix:
                paint(self.tbl_res.item(i,1), mix)
            else:
                self.tbl_res.item(i,1).setText("")
                self.tbl_res.item(i,1).setBackground(_brush(GRAY_RES))
            paint(self.tbl_res.item(i,2), vap)
            paint(self.tbl_res.item(i,3), liq)

        # ── Composiciones ─────────────────────────────────────
        sy = sx = 0
        self.tbl_comp.blockSignals(True)
        for i in range(NC):
            yi = y[i] if i < len(y) else 0
            xi = x[i] if i < len(x) else 0
            if masa:
                yi_s = yi*PM[i]/PM_v if (PM_v and PM_v>0) else 0
                xi_s = xi*PM[i]/PM_l if (PM_l and PM_l>0) else 0
            else:
                yi_s, xi_s = yi, xi
            sy += yi_s; sx += xi_s
            tv = f"{yi_s:.4f}" if V>0 else ""
            tl = f"{xi_s:.4f}" if L>0 else ""
            it2 = self.tbl_comp.item(i,2)
            it3 = self.tbl_comp.item(i,3)
            it2.setText(tv); it2.setBackground(_brush(WHITE if tv else GRAY_RES))
            it3.setText(tl); it3.setBackground(_brush(WHITE if tl else GRAY_RES))
        self.tbl_comp.blockSignals(False)

        ts2 = f"{sy:.4f}" if V>0 else ""
        ts3 = f"{sx:.4f}" if L>0 else ""
        self.tbl_comp.item(self.sum_row,2).setText(ts2)
        self.tbl_comp.item(self.sum_row,3).setText(ts3)
        self.tbl_comp.item(self.sum_row,2).setBackground(_brush(WHITE if ts2 else GRAY_RES))
        self.tbl_comp.item(self.sum_row,3).setBackground(_brush(WHITE if ts3 else GRAY_RES))


# ══════════════════════════════════════════════════════════════
# Tab 2 — Parámetros EOS
# ══════════════════════════════════════════════════════════════
class TabParametros(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4,4,4,4)
        outer.setSpacing(3)
        self.setStyleSheet(f'background:{GRAY_LBL};')

        WP = [200,160,150,145,165]
        WK = 65

        # ─── Tabla propiedades críticas (título+cabecera+datos en una sola tabla) ─
        outer.addWidget(title_label("Propiedades criticas y factor acentrico"))

        self.tbl_p = QTableWidget(NC+1, 5)  # fila 0=cabecera, filas 1..NC=datos
        self.tbl_p.horizontalHeader().hide()
        self.tbl_p.verticalHeader().hide()
        self.tbl_p.setShowGrid(True)
        self.tbl_p.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_p.setStyleSheet(
            f'QTableWidget {{ border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;gridline-color:{BORDER};}}'
            f'QTableWidget::item {{ padding:2px 6px; }}')
        for c,w in enumerate(WP): self.tbl_p.setColumnWidth(c,w)
        for r in range(NC+1): self.tbl_p.setRowHeight(r, ROW_H)

        # Fila 0: cabecera (se desplaza con scroll)
        for c,h in enumerate(["Componente","Temperatura Critica (°R)",
                               "Presion Critica (psi)","Factor acentrico",
                               "Peso Molecular (lb/lb-mol)"]):
            self.tbl_p.setItem(0,c, cell(h, bg=GRAY_LBL,
                align=Qt.AlignmentFlag.AlignCenter))

        # Filas 1..NC: datos
        for i in range(NC):
            r = i+1
            self.tbl_p.setItem(r,0, cell(NOMBRES[i], bg=GRAY_LBL,
                align=Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter))
            for c,v in enumerate([f"{TC[i]:.4f}",f"{PC[i]:.4f}",
                                   f"{OMEGA[i]:.8f}",f"{PM[i]}"]):
                self.tbl_p.setItem(r,c+1, cell(v, bg=WHITE, color=TEXT_RES))

        self.tbl_p.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl_p.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl_p.setFixedHeight(309)
        outer.addWidget(self.tbl_p)  # altura fija 310px

        # ─── Tabla kij (cabecera+datos en una sola tabla) ─────
        outer.addWidget(title_label("Coeficientes de interaccion binaria"))

        self.tbl_k = QTableWidget(NC+1, NC+1)  # fila 0=cabecera
        self.tbl_k.horizontalHeader().hide()
        self.tbl_k.verticalHeader().hide()
        self.tbl_k.setShowGrid(True)
        self.tbl_k.setStyleSheet(
            f'QTableWidget {{ border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;gridline-color:{BORDER};}}'
            f'QTableWidget::item {{ padding:2px 4px; }}')
        self.tbl_k.setColumnWidth(0, WK)
        for c in range(1,NC+1): self.tbl_k.setColumnWidth(c, WK)
        for r in range(NC+1): self.tbl_k.setRowHeight(r, ROW_H)

        # Fila 0: cabecera (se desplaza con scroll)
        self.tbl_k.setItem(0,0, cell("", bg=GRAY_LBL))
        for j,comp in enumerate(COMPONENTES):
            self.tbl_k.setItem(0,j+1, cell(comp, bg=GRAY_LBL,
                align=Qt.AlignmentFlag.AlignCenter))

        # Filas 1..NC: datos
        for i in range(NC):
            r = i+1
            self.tbl_k.setItem(r,0, cell(COMPONENTES[i], bg=GRAY_LBL,
                align=Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter))
            for j in range(NC):
                v = kij_user[i][j]
                if i == j:
                    it = cell(f"{v:.5f}", bg=GRAY_LBL,
                        color=TEXT_DIM, align=Qt.AlignmentFlag.AlignCenter)
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    it = cell(f"{v:.5f}", bg=WHITE, color=TEXT_RES,
                        align=Qt.AlignmentFlag.AlignCenter, editable=True)
                self.tbl_k.setItem(r,j+1, it)

        self.tbl_k.itemChanged.connect(self._on_kij)
        self.tbl_k.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl_k.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl_k.setFixedHeight(310)
        outer.addWidget(self.tbl_k)  # altura fija 316px

        bot = QHBoxLayout()
        note = QLabel("Doble clic para editar un coeficiente "
                      "(la celda simetrica se actualiza automaticamente)")
        note.setStyleSheet(
            f'color:{TEXT_DIM};font-size:9pt;font-family:"{FONT_F}";'
            f'background:transparent;')
        bot.addWidget(note)
        bot.addStretch()
        btn_r = QPushButton("Restaurar valores originales")
        btn_r.setFixedWidth(220)
        btn_r.setStyleSheet(
            f'background:{GRAY_LBL};border:2px outset {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;min-height:22px;')
        btn_r.clicked.connect(self._reset)
        bot.addWidget(btn_r)
        outer.addLayout(bot)

    def _on_kij(self, item):
        r = item.row(); c = item.column()
        if r < 1 or c < 1: return   # fila 0 = cabecera
        i = r-1; j = c-1
        if i == j: return
        try:
            v = float(item.text())
            kij_user[i][j] = v
            kij_user[j][i] = v
            self.tbl_k.blockSignals(True)
            sym = self.tbl_k.item(j+1, i+1)  # +1 por fila de cabecera
            if sym: sym.setText(f"{v:.5f}")
            item.setBackground(_brush(WHITE))
            item.setForeground(_brush(TEXT_RES))
            self.tbl_k.blockSignals(False)
        except: pass

    def _reset(self):
        global kij_user
        kij_user = copy.deepcopy(KIJ_DEFAULT)
        self.tbl_k.blockSignals(True)
        for i in range(NC):
            for j in range(NC):
                it = self.tbl_k.item(i+1, j+1)  # +1 por fila de cabecera
                if it and i != j:
                    it.setText(f"{kij_user[i][j]:.5f}")
        self.tbl_k.blockSignals(False)
        QMessageBox.information(self, "Coeficientes", "Coeficientes restaurados.")



# ══════════════════════════════════════════════════════════════
# Pantalla de Carga (Splash Screen)
# ══════════════════════════════════════════════════════════════
class SplashScreen(QWidget):
    """Pantalla de carga mostrada mientras ThermoPhase inicia."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(340, 269)
        self._img = None
        # Cargar imagen splash
        import sys as _s, os as _o
        _base = getattr(_s, '_MEIPASS', _o.path.dirname(_o.path.abspath(__file__)))
        _sp = _o.path.join(_base, 'splash.png')
        if _o.path.exists(_sp):
            from PyQt6.QtGui import QPixmap
            self._img = QPixmap(_sp)
        # Centrar en pantalla
        from PyQt6.QtWidgets import QApplication
        sg = QApplication.primaryScreen().geometry()
        self.move((sg.width()-340)//2, (sg.height()-269)//2)

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QPen, QFont
        from PyQt6.QtCore import Qt as _Qt
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._img:
            p.drawPixmap(0, 0, self._img)
        else:
            # Fallback: rectángulo oscuro con texto
            p.setBrush(QColor(20,10,5))
            p.setPen(_Qt.PenStyle.NoPen)
            p.drawRoundedRect(0,0,420,260,16,16)
            p.setPen(QColor(240,144,48))
            fnt = QFont("Arial Narrow", 28, QFont.Weight.Bold)
            p.setFont(fnt)
            p.drawText(50,80,320,60, _Qt.AlignmentFlag.AlignCenter, "ThermoPhase")
            p.setPen(QColor(200,160,120))
            fnt2 = QFont("Arial Narrow", 11)
            p.setFont(fnt2)
            p.drawText(50,140,320,40, _Qt.AlignmentFlag.AlignCenter,
                       "Calculadora de Equilibrio de Fases")
        p.end()

# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThermoPhase")
        TW = W_COMP + 3*W_VAL + 30   # ancho total + márgenes
        TH = 805
        self.setFixedSize(TW, TH)
        self._build()

    def _build(self):
        cw = QWidget(); self.setCentralWidget(cw)
        lay = QVBoxLayout(cw)
        lay.setContentsMargins(4,4,4,2); lay.setSpacing(2)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            f'QTabWidget::pane {{border:1px solid {BORDER};}}'
            f'QTabBar::tab {{background:{GRAY_LBL};color:{TEXT};'
            f'padding:4px 14px;border:1px solid {BORDER};border-bottom:none;'
            f'margin-right:1px;font-family:"{FONT_F}";font-size:{FS}pt;}}'
            f'QTabBar::tab:selected {{background:{WHITE};'
            f'border-bottom:1px solid {WHITE};}}'
        )
        self.tab_eq  = TabEquilibrio()
        self.tab_env = TabEnvolvente(
            get_z=self.tab_eq.get_z,
            get_kij=lambda: kij_user
        )
        self.tab_par = TabParametros()
        tabs.addTab(self.tab_eq,  "Equilibrio de fases")
        tabs.addTab(self.tab_env, "Envolvente de fases")
        tabs.addTab(self.tab_par, "Parametros de la ecuacion de estado")
        lay.addWidget(tabs)

        sb = QStatusBar()
        sb.setStyleSheet(
            f'background:{GRAY_LBL};font-family:"{FONT_F}";font-size:9pt;'
            f'border-top:1px solid {BORDER};')
        sb.showMessage(
            f"  Peng-Robinson EOS  |  "
            f"R = {R_GAS} psi·ft³/(lb-mol·°R)  |  13 componentes")
        self.setStatusBar(sb)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Ícono global
    import sys as _sys2, time as _time
    _base2 = getattr(_sys2, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    _ico2  = os.path.join(_base2, 'thermophase.ico')
    if os.path.exists(_ico2):
        app.setWindowIcon(QIcon(_ico2))
    # Splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    _t_ini = _time.time()
    # Cargar ventana principal
    win = MainWindow()
    # Mantener el splash visible al menos 2 segundos en total
    _espera = 2.0 - (_time.time() - _t_ini)
    if _espera > 0:
        _time.sleep(_espera)
    splash.close()
    win.show()
    sys.exit(app.exec())
