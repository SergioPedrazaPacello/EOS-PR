"""
Pestaña Puntos de Saturación para ThermoPhase.
Calcula T de rocío, T de burbuja, P de rocío, P de burbuja.
Mismo estilo (Arial Narrow) que el resto del programa.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QGridLayout, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSizePolicy, QAbstractSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QBrush

from engine3 import NOMBRES, NC

WHITE="#FFFFFF"; GRAY_TIT="#A8A8A8"; GRAY_HDR="#C8C8C8"; GRAY_LBL="#D0D0D0"
GRAY_RES="#E8E8E8"; BORDER="#888888"; TEXT="#000000"; TEXT_DIM="#555555"
TEXT_RES="#000080"; FONT_F="Arial Narrow"; FS=10

BTN_STYLE=(f'background:{GRAY_LBL};border:2px outset {BORDER};'
           f'font-family:"{FONT_F}";font-size:{FS}pt;min-height:22px;')
LBL_TIT=(f'background:{GRAY_TIT};color:{TEXT};border:1px solid {BORDER};'
         f'font-family:"{FONT_F}";font-size:{FS}pt;padding:0px 6px;')
LBL_SEC=(f'background:{GRAY_LBL};color:{TEXT};border:1px solid {BORDER};'
         f'font-family:"{FONT_F}";font-size:{FS}pt;padding:0px 6px;')
LBL_RES=(f'background:{GRAY_LBL};border:1px solid {BORDER};color:{TEXT_RES};'
         f'font-family:"{FONT_F}";font-size:{FS}pt;padding:2px 6px;')


# ── Worker para cálculo en segundo plano ──────────────────────
class SatWorker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, tipo, valor, z, kij):
        super().__init__()
        self.tipo=tipo; self.valor=valor; self.z=z; self.kij=kij
    def run(self):
        try:
            from envelope import punto_saturacion
            res = punto_saturacion(self.tipo, self.valor, self.z, self.kij)
            self.done.emit(res if res else {})
        except Exception as e:
            self.error.emit(str(e))


class TabSaturacion(QWidget):
    # Mapeo desplegable → (tipo_calc, unidad_entrada, etiqueta_entrada, unidad_result)
    TIPOS = {
        "Temperatura de Rocío":   ('T_rocio',   'P', 'Presion (psi):',      'T'),
        "Temperatura de Burbuja": ('T_burbuja', 'P', 'Presion (psi):',      'T'),
        "Presion de Rocío":       ('P_rocio',   'T', 'Temperatura (°R):',   'P'),
        "Presion de Burbuja":     ('P_burbuja', 'T', 'Temperatura (°R):',   'P'),
    }

    def __init__(self, get_z, get_kij):
        super().__init__()
        self.get_z=get_z; self.get_kij=get_kij
        self.worker=None
        self._build()

    def _build(self):
        self.setStyleSheet(f'background:{GRAY_LBL};')
        root=QVBoxLayout(self)
        root.setContentsMargins(4,10,4,4); root.setSpacing(3)

        # Título
        title=QLabel("ThermoPhase — Puntos de Saturación")
        title.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        title.setFixedHeight(22); title.setStyleSheet(LBL_TIT)
        root.addWidget(title)

        # ── Panel de entrada ──────────────────────────────────
        in_box=QFrame()
        in_box.setStyleSheet('background:transparent;border:none;')
        gl=QGridLayout(in_box); gl.setContentsMargins(6,6,6,6); gl.setSpacing(6)

        def lbl(txt, res=False):
            l=QLabel(txt)
            if res:
                l.setStyleSheet(
                    f'background:transparent;border:1px solid {BORDER};'
                    f'color:{TEXT_RES};padding:2px 6px;'
                    f'font-family:"{FONT_F}";font-size:{FS}pt;')
            else:
                l.setStyleSheet(
                    f'background:transparent;border:1px solid {BORDER};'
                    f'padding:2px 6px;font-family:"{FONT_F}";font-size:{FS}pt;')
            l.setFixedHeight(24)
            return l

        # Selector de tipo de cálculo
        gl.addWidget(lbl("Calcular:"), 0, 0)
        self.cmb_tipo=QComboBox()
        self.cmb_tipo.addItems(list(self.TIPOS.keys()))
        self.cmb_tipo.setFixedHeight(24); self.cmb_tipo.setFixedWidth(160)
        self.cmb_tipo.setStyleSheet(
            f'QComboBox {{ background:{WHITE};border:1px solid {BORDER};'
            f'color:{TEXT};font-family:"{FONT_F}";font-size:{FS}pt; padding:1px 4px; }}')
        self.cmb_tipo.currentTextChanged.connect(self._on_tipo_change)
        gl.addWidget(self.cmb_tipo, 0, 1)

        # Etiqueta + campo de condición (P o T)
        self.lbl_cond=lbl("Presion (psi):")
        self.lbl_cond.setFixedWidth(130)
        gl.addWidget(self.lbl_cond, 1, 0)
        self.sp_cond=QDoubleSpinBox()
        self.sp_cond.setRange(0.0, 15000.0); self.sp_cond.setDecimals(2)
        self.sp_cond.setSpecialValueText(" ")   # muestra vacío en el mínimo
        self.sp_cond.setValue(0.0)              # inicia vacío
        self.sp_cond.setFixedHeight(24)
        self.sp_cond.setFixedWidth(160)
        # Sin flechas de incremento/decremento
        self.sp_cond.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sp_cond.setStyleSheet(
            f'QDoubleSpinBox {{ background:{WHITE};border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt; }}')
        gl.addWidget(self.sp_cond, 1, 1)

        # Botón calcular
        self.btn=QPushButton("Calcular punto de saturacion")
        self.btn.setStyleSheet(BTN_STYLE); self.btn.setFixedHeight(26)
        self.btn.clicked.connect(self.calcular)
        gl.addWidget(self.btn, 2, 0, 1, 2)

        gl.setColumnStretch(0,0); gl.setColumnStretch(1,1)
        in_box.setMaximumWidth(360)
        root.addWidget(in_box, alignment=Qt.AlignmentFlag.AlignLeft)

        # ── Panel de resultados ───────────────────────────────
        res_title=QLabel("Resultado:")
        res_title.setStyleSheet(LBL_SEC); res_title.setFixedHeight(22)
        root.addWidget(res_title)

        res_box=QFrame()
        res_box.setStyleSheet('background:transparent;border:none;')
        rl=QGridLayout(res_box); rl.setContentsMargins(6,4,6,4); rl.setSpacing(4)

        self.lbl_res_label=lbl("Temperatura de rocio (°F):")
        self.lbl_res_label.setFixedWidth(200)
        rl.addWidget(self.lbl_res_label, 0, 0)
        self.lbl_res_val=lbl("", res=True)
        self.lbl_res_val.setFixedWidth(120)
        self.lbl_res_val.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        rl.addWidget(self.lbl_res_val, 0, 1)

        self.lbl_res2_label=lbl("Equivalente (°R / psi):")
        self.lbl_res2_label.setFixedWidth(200)
        rl.addWidget(self.lbl_res2_label, 1, 0)
        self.lbl_res2_val=lbl("", res=True)
        self.lbl_res2_val.setFixedWidth(120)
        self.lbl_res2_val.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        rl.addWidget(self.lbl_res2_val, 1, 1)

        self.lbl_estado=QLabel("")
        self.lbl_estado.setStyleSheet(
            f'color:{TEXT_DIM};font-family:"{FONT_F}";font-size:9pt;background:transparent;')
        rl.addWidget(self.lbl_estado, 2, 0, 1, 2)

        rl.setColumnStretch(0,0); rl.setColumnStretch(1,1)
        res_box.setMaximumWidth(360)
        root.addWidget(res_box, alignment=Qt.AlignmentFlag.AlignLeft)

        # ── Tabla de composiciones de las fases ───────────────
        comp_title=QLabel("Composicion de las fases en equilibrio:")
        comp_title.setStyleSheet(LBL_SEC); comp_title.setFixedHeight(22)
        root.addWidget(comp_title)

        self.tbl=QTableWidget(NC+1, 3)
        self.tbl.setHorizontalHeaderLabels(["Componente","Fase Vapor","Fase Liquida"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tbl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tbl.setStyleSheet(
            f'QTableWidget {{ border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;gridline-color:{BORDER};}}'
            f'QHeaderView::section {{ background:{GRAY_HDR};border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;padding:2px; }}')
        hh=self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tbl.setColumnWidth(1,130); self.tbl.setColumnWidth(2,130)
        self.tbl.verticalHeader().setDefaultSectionSize(20)
        self.tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        GRIS_NOMBRE = QColor("#E8E8E8")   # gris claro para nombres
        for i in range(NC):
            it=QTableWidgetItem(NOMBRES[i].rstrip(':'))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            it.setBackground(QBrush(GRIS_NOMBRE))
            self.tbl.setItem(i,0,it)
            for c in (1,2):
                cell=QTableWidgetItem("")
                cell.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
                self.tbl.setItem(i,c,cell)
        # Fila sumatorias
        sit=QTableWidgetItem("Sumatorias:")
        sit.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        sit.setBackground(QBrush(GRIS_NOMBRE))
        self.tbl.setItem(NC,0,sit)
        for c in (1,2):
            cell=QTableWidgetItem("")
            cell.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            self.tbl.setItem(NC,c,cell)

        root.addWidget(self.tbl, stretch=1)

        # ── Panel de propiedades del punto de saturación ──────
        prop_title=QLabel("Propiedades del punto de saturacion:")
        prop_title.setStyleSheet(LBL_SEC); prop_title.setFixedHeight(22)
        root.addWidget(prop_title)

        self.tbl_prop=QTableWidget(6, 3)
        self.tbl_prop.setHorizontalHeaderLabels(
            ["Propiedad","Fase Vapor","Fase Liquida"])
        self.tbl_prop.verticalHeader().setVisible(False)
        self.tbl_prop.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_prop.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tbl_prop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tbl_prop.setStyleSheet(
            f'QTableWidget {{ border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;gridline-color:{BORDER};}}'
            f'QHeaderView::section {{ background:{GRAY_HDR};border:1px solid {BORDER};'
            f'font-family:"{FONT_F}";font-size:{FS}pt;padding:2px; }}')
        hp=self.tbl_prop.horizontalHeader()
        hp.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hp.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hp.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tbl_prop.setColumnWidth(1,130); self.tbl_prop.setColumnWidth(2,130)
        self.tbl_prop.verticalHeader().setDefaultSectionSize(20)
        self.tbl_prop.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl_prop.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        _props=["Peso molecular","Factor de compresibilidad",
                "Densidad masica [lb/ft3]","Gravedad especifica"]
        GRIS=QColor("#E8E8E8")
        for r,lbl_p in enumerate(_props):
            it=QTableWidgetItem(lbl_p)
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            it.setBackground(QBrush(GRIS))
            self.tbl_prop.setItem(r,0,it)
            for c in (1,2):
                cc=QTableWidgetItem("")
                cc.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
                self.tbl_prop.setItem(r,c,cc)
        self.tbl_prop.setRowCount(4)
        self.tbl_prop.setMaximumHeight(4*20 + 26)
        root.addWidget(self.tbl_prop)

    def _on_tipo_change(self, txt):
        tipo, unidad, etiqueta, _ = self.TIPOS[txt]
        self.lbl_cond.setText(etiqueta)
        if unidad=='P':
            self.sp_cond.setRange(0.0, 15000.0)
        else:
            self.sp_cond.setRange(0.0, 2000.0)
        # No forzar valor — dejar lo que el usuario haya puesto o vacío

    def calcular(self):
        z=self.get_z()
        if abs(sum(z)-1.0)>1e-3:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self,"Composición",
                "La suma de fracciones debe ser 1.0")
            return
        kij=self.get_kij()
        txt=self.cmb_tipo.currentText()
        tipo, unidad, etiqueta, res_unit = self.TIPOS[txt]
        valor=self.sp_cond.value()
        if valor <= 0.0:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self,"Dato faltante",
                "Ingrese un valor de presion o temperatura.")
            return

        self.btn.setEnabled(False); self.btn.setText("Calculando...")
        self.lbl_estado.setText("")
        self._res_unit=res_unit; self._tipo_txt=txt
        self.worker=SatWorker(tipo, valor, z, kij)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_error(self, msg):
        self.btn.setEnabled(True); self.btn.setText("Calcular punto de saturacion")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self,"Error",msg)

    def _on_done(self, res):
        self.btn.setEnabled(True); self.btn.setText("Calcular punto de saturacion")
        if not res or not res.get('exito'):
            self.lbl_estado.setText("No se encontro punto de saturacion")
            self.lbl_res_val.setText(""); self.lbl_res2_val.setText("")
            return

        T=res['T']; P=res['P']
        if self._res_unit=='T':
            self.lbl_res_label.setText(f"{self._tipo_txt} (°F):")
            self.lbl_res_val.setText(f"{T-459.67:.2f}")
            self.lbl_res2_label.setText("Equivalente (°R):")
            self.lbl_res2_val.setText(f"{T:.2f}")
        else:
            self.lbl_res_label.setText(f"{self._tipo_txt} (psi):")
            self.lbl_res_val.setText(f"{P:.2f}")
            self.lbl_res2_label.setText("Temperatura (°F):")
            self.lbl_res2_val.setText(f"{T-459.67:.2f}")

        self.lbl_estado.setText("Convergencia exitosa.")

        # Llenar tabla de composiciones
        x=res.get('x',[0]*NC); y=res.get('y',[0]*NC)
        sx=sum(x); sy=sum(y)
        for i in range(NC):
            self.tbl.item(i,1).setText(f"{y[i]:.4f}")
            self.tbl.item(i,2).setText(f"{x[i]:.4f}")
            self.tbl.item(i,1).setBackground(QBrush(QColor(WHITE)))
            self.tbl.item(i,2).setBackground(QBrush(QColor(WHITE)))
            self.tbl.item(i,1).setForeground(QBrush(QColor(TEXT_RES)))
            self.tbl.item(i,2).setForeground(QBrush(QColor(TEXT_RES)))
        self.tbl.item(NC,1).setText(f"{sy:.4f}")
        self.tbl.item(NC,2).setText(f"{sx:.4f}")

        # Llenar panel de propiedades
        p=res.get('props',{})
        def setp(row, key_v, key_l, fmt="{:.4f}"):
            vv=p.get(key_v); vl=p.get(key_l)
            self.tbl_prop.item(row,1).setText(fmt.format(vv) if vv is not None else "")
            self.tbl_prop.item(row,2).setText(fmt.format(vl) if vl is not None else "")
            self.tbl_prop.item(row,1).setForeground(QBrush(QColor(TEXT_RES)))
            self.tbl_prop.item(row,2).setForeground(QBrush(QColor(TEXT_RES)))
        setp(0,'PM_v','PM_l')
        setp(1,'ZV','ZL')
        setp(2,'rho_v','rho_l')
        setp(3,'sg_v','sg_l')
