"""
Pestaña Envolvente de Fases para ThermoPhase.
Mismo estilo (Arial Narrow) que el resto del programa.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QProgressBar, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib import font_manager

# Colores (mismos que app_final.py)
WHITE="#FFFFFF"; GRAY_TIT="#A8A8A8"; GRAY_HDR="#C8C8C8"; GRAY_LBL="#D0D0D0"; GRAY_RES="#E8E8E8"
BORDER="#888888"; TEXT="#000000"; TEXT_DIM="#555555"; TEXT_RES="#000080"
FONT_F="Arial Narrow"; FS=10

# Configurar matplotlib para usar Arial Narrow
matplotlib.rcParams['font.family'] = ['Arial Narrow', 'Arial', 'sans-serif']

BTN_STYLE=(f'background:{GRAY_HDR};border:2px outset {BORDER};'
           f'font-family:"{FONT_F}";font-size:{FS}pt;min-height:22px;')
LBL_HDR=(f'background:{GRAY_TIT};color:{TEXT};border:1px solid {BORDER};'
         f'font-family:"{FONT_F}";font-size:{FS}pt;padding:0px 6px;')
LBL_SEC=(f'background:{GRAY_LBL};color:{TEXT};border:1px solid {BORDER};'
         f'font-family:"{FONT_F}";font-size:{FS}pt;padding:0px 6px;')
LBL_RES=(f'background:{GRAY_RES};border:1px solid {BORDER};'
         f'font-family:"{FONT_F}";font-size:{FS}pt;padding:2px 6px;color:{TEXT_RES};')


class EnvWorker(QThread):
    done=pyqtSignal(dict); error=pyqtSignal(str)
    def __init__(self, z, kij):
        super().__init__(); self.z=z; self.kij=kij
    def run(self):
        try:
            from envelope import curva_envolvente
            res=curva_envolvente(self.z, self.kij)
            self.done.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class TabEnvolvente(QWidget):
    def __init__(self, get_z, get_kij):
        super().__init__()
        self.get_z=get_z; self.get_kij=get_kij
        self.worker=None; self.result=None
        self._build()

    def _build(self):
        self.setObjectName('envTab')
        self.setStyleSheet(f'QWidget#envTab {{ background:{GRAY_LBL}; }}')
        root=QVBoxLayout(self)
        root.setContentsMargins(4,10,4,4); root.setSpacing(3)

        title=QLabel("ThermoPhase — Envolvente de Fases")
        title.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        title.setFixedHeight(22); title.setStyleSheet(LBL_HDR)
        root.addWidget(title)

        content=QHBoxLayout()
        content.setContentsMargins(6,4,6,4); content.setSpacing(8)

        # Contenedor izquierdo: placeholder (vacío) o canvas (con datos)
        self.left_box = QWidget()
        self.left_box.setStyleSheet(f'background:{WHITE};border:1px solid {BORDER};')
        self.left_box.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)
        left_lay = QVBoxLayout(self.left_box)
        left_lay.setContentsMargins(0,0,0,0); left_lay.setSpacing(0)

        # Gráfico (oculto al inicio)
        self.fig=Figure(figsize=(1,1))
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ax=self.fig.add_subplot(111)
        self.ax.set_position([0.14, 0.09, 0.83, 0.88])
        self.canvas=FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setVisible(False)   # oculto hasta calcular
        left_lay.addWidget(self.canvas)

        content.addWidget(self.left_box, stretch=1)

        # Panel derecho
        right=QWidget(); right.setFixedWidth(210)
        vr=QVBoxLayout(right); vr.setContentsMargins(0,0,0,0); vr.setSpacing(6)

        self.btn=QPushButton("Calcular Envolvente")
        self.btn.setStyleSheet(BTN_STYLE); self.btn.setFixedHeight(30)
        self.btn.clicked.connect(self.calcular)
        vr.addWidget(self.btn)

        # Barra de progreso estándar
        self.prog=QProgressBar()
        self.prog.setRange(0,0)            # modo indeterminado
        self.prog.setVisible(False)
        self.prog.setTextVisible(False)    # sin texto
        self.prog.setFixedHeight(18)
        self.prog.setStyleSheet(
            f'QProgressBar {{ border:1px solid #888888; background:#E8E8E8;'
            f'border-radius:0px; }}'
            f'QProgressBar::chunk {{ background:#2d7d2d; }}')
        vr.addWidget(self.prog)

        sep=QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BORDER};')
        vr.addWidget(sep)

        res_title=QLabel("Puntos especiales:")
        res_title.setStyleSheet(LBL_SEC); res_title.setFixedHeight(22)
        vr.addWidget(res_title)

        grid=QGridLayout(); grid.setSpacing(4); grid.setContentsMargins(0,2,0,0)
        lbl_style=(f'font-family:"{FONT_F}";font-size:{FS}pt;'
                   f'color:{TEXT};background:transparent;')

        def res_val():
            l=QLabel("—"); l.setStyleSheet(LBL_RES)
            l.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            return l

        rows=[("Cricondentérmica (°F):","cric_T"),
              ("Cricondenbárica (psi):","cric_P")]
        self.res_labels={}
        for r,(txt,key) in enumerate(rows):
            lbl=QLabel(txt); lbl.setStyleSheet(lbl_style); lbl.setWordWrap(True)
            grid.addWidget(lbl,r,0)
            rv=res_val(); self.res_labels[key]=rv
            grid.addWidget(rv,r,1)
        vr.addLayout(grid)
        vr.addStretch()

        self.btn_exp=QPushButton("Exportar CSV")
        self.btn_exp.setStyleSheet(BTN_STYLE); self.btn_exp.setEnabled(False)
        self.btn_exp.clicked.connect(self.exportar_csv)
        vr.addWidget(self.btn_exp)

        content.addWidget(right)
        root.addLayout(content, stretch=1)


    def calcular(self):
        z=self.get_z()
        if abs(sum(z)-1.0)>1e-3:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self,"Composición",
                "La suma de fracciones debe ser 1.0")
            return
        kij=self.get_kij()
        self.btn.setEnabled(False); self.btn.setText("Calculando...")
        self.prog.setVisible(True)
        self.worker=EnvWorker(z,kij)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_error(self,msg):
        self.btn.setEnabled(True); self.btn.setText("Calcular Envolvente")
        self.prog.setVisible(False)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self,"Error",msg)

    def _on_done(self,res):
        self.btn.setEnabled(True); self.btn.setText("Calcular Envolvente")
        self.prog.setVisible(False)
        self.result=res
        self.canvas.setVisible(True)   # mostrar el gráfico ya con datos
        self._plot(res)
        self._update_results(res)
        self.btn_exp.setEnabled(True)

    def _plot(self,res):
        ax=self.ax; ax.clear()
        ax.set_facecolor('#FAFAFA')
        burb=res.get('burbuja',[]); rocio=res.get('rocio',[])
        Tb=[t-459.67 for _,t in burb]; Pb=[p for p,_ in burb]
        Td=[t-459.67 for _,t in rocio]; Pd=[p for p,_ in rocio]

        if Tb and Pb:
            ax.plot(Tb,Pb,linestyle='none',marker='^',
                    color='#a83218',markersize=3,
                    label='Curva de Burbuja')
        if Td and Pd:
            ax.plot(Td,Pd,linestyle='none',marker='^',
                    color='#1a4fa8',markersize=3,
                    label='Curva de Rocío')

        ax.set_xlabel("Temperatura (°F)", fontsize=9, color=TEXT_DIM)
        ax.set_ylabel("Presión (psia)", fontsize=9, color=TEXT_DIM)

        ax.tick_params(labelsize=8, colors=TEXT_DIM)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)
        ax.grid(True, linestyle='--', alpha=0.4, color=GRAY_LBL)
        if Tb or Td:
            ax.legend(fontsize=8, framealpha=0.9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.set_position([0.14, 0.09, 0.83, 0.88])
        self.canvas.draw_idle()

    def _update_results(self,res):
        burb=res.get('burbuja',[]); rocio=res.get('rocio',[])
        def fv(v): return f"{v:.1f}" if v is not None else "—"
        Tb=[t-459.67 for _,t in burb]; Pb=[p for p,_ in burb]
        Td=[t-459.67 for _,t in rocio]; Pd=[p for p,_ in rocio]
        all_T=Tb+Td; all_P=Pb+Pd
        # Cricondentérmica = T máxima de la envolvente
        self.res_labels['cric_T'].setText(fv(max(all_T)) if all_T else "—")
        # Cricondenbárica = P máxima de la envolvente
        self.res_labels['cric_P'].setText(fv(max(all_P)) if all_P else "—")

    def exportar_csv(self):
        if not self.result: return
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        path,_=QFileDialog.getSaveFileName(self,"Guardar CSV",
            "envolvente.csv","CSV (*.csv)")
        if not path: return
        try:
            with open(path,'w',encoding='utf-8') as f:
                f.write("Curva,P (psia),T (R),T (F)\n")
                for p,t in self.result.get('burbuja',[]):
                    f.write(f"Burbuja,{p:.4f},{t:.4f},{t-459.67:.4f}\n")
                for p,t in self.result.get('rocio',[]):
                    f.write(f"Rocio,{p:.4f},{t:.4f},{t-459.67:.4f}\n")
            QMessageBox.information(self,"Exportar",f"CSV guardado:\n{path}")
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))
