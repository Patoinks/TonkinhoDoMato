"""Native local dashboard for Tonkinho do Mato training."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

from tracker_data import TrainingReader

BG = '#101813'
CARD = '#19251e'
TEXT = '#eef4e9'
MUTED = '#9caf9f'
GREEN = '#b8e77a'
BLUE = '#81c7d7'
RED = '#f19687'
GOLD = '#e6cb82'


def number(value):
    return f'{int(value):,}'.replace(',', ' ')


def eta(seconds):
    if seconds is None:
        return '—'
    minutes = max(1, round(seconds / 60))
    return f'{minutes // 60} h {minutes % 60:02d} min' if minutes >= 60 else f'{minutes} min'


class Chart(tk.Canvas):
    def __init__(self, parent, title, unit, colour):
        super().__init__(parent, bg=CARD, highlightthickness=0, height=190)
        self.title, self.unit, self.colour, self.points = title, unit, colour, []
        self.bind('<Configure>', lambda _: self.render())

    def render(self, points=None):
        if points is not None:
            self.points = points
        self.delete('all')
        width, height = self.winfo_width(), self.winfo_height()
        self.create_text(18, 20, text=self.title, fill=TEXT, anchor='w', font=('Segoe UI', 11, 'bold'))
        self.create_text(width-18, 20, text=self.unit, fill=MUTED, anchor='e', font=('Segoe UI', 9))
        if len(self.points) < 2:
            self.create_text(width/2, height/2, text='A recolher dados…', fill=MUTED, font=('Segoe UI', 11))
            return
        data = self.points[-160:]
        high = max(y for _, y in data) * 1.12 or 1
        left, right, top, bottom = 46, width-20, 46, height-33
        for i in range(4):
            y = top + (bottom-top)*i/3
            self.create_line(left, y, right, y, fill='#2b3b30')
            self.create_text(left-8, y, text=f'{high*(1-i/3):.1f}', anchor='e', fill=MUTED, font=('Segoe UI', 8))
        low_x, high_x = data[0][0], data[-1][0]
        line = []
        for x, y in data:
            line.extend((left+(right-left)*(x-low_x)/max(1,high_x-low_x), bottom-(bottom-top)*y/high))
        self.create_polygon(left, bottom, *line, right, bottom, fill='#23382a' if self.colour == GREEN else '#20333a', outline='')
        self.create_line(*line, fill=self.colour, width=2)
        self.create_oval(line[-2]-3, line[-1]-3, line[-2]+3, line[-1]+3, fill=self.colour, outline='')
        self.create_text(left, height-14, text=f'Jogo {number(low_x)}', anchor='w', fill=MUTED, font=('Segoe UI', 8))
        self.create_text(right, height-14, text=number(high_x), anchor='e', fill=MUTED, font=('Segoe UI', 8))


class Tracker:
    def __init__(self, root, project):
        self.root, self.project = root, Path(project)
        self.reader = TrainingReader(project)
        self.snapshot = None
        root.title('Tonkinho do Mato · Treino')
        root.configure(bg=BG)
        height = min(940, root.winfo_screenheight()-80)
        if height < 900:
            root.tk.call('tk', 'scaling', 1.2)
        root.geometry(f'1120x{height}')
        root.minsize(940, min(900,height))
        root.option_add('*Font', '{Segoe UI} 10')
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1, minsize=180)
        header = tk.Frame(root, bg=BG)
        header.grid(row=0, column=0, sticky='ew', padx=28, pady=(24,18))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text='TONKINHO DO MATO', bg=BG, fg=GREEN, font=('Segoe UI', 10, 'bold')).grid(row=0,column=0,sticky='w')
        tk.Label(header, text='O treino, em direto.', bg=BG, fg=TEXT, font=('Segoe UI', 25, 'bold')).grid(row=1,column=0,sticky='w')
        self.status = tk.Label(header, text='A ligar aos logs…', bg=CARD, fg=GREEN, padx=16, pady=9)
        self.status.grid(row=0,column=1,rowspan=2,sticky='e')
        asset = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'tonkinho-do-mato.png'
        if asset.exists():
            self.icon = tk.PhotoImage(file=str(asset))
            root.iconphoto(True, self.icon)
        progress = tk.Frame(root, bg=CARD, padx=20, pady=16)
        progress.grid(row=1,column=0,sticky='ew',padx=28)
        progress.grid_columnconfigure(0,weight=1)
        self.opponent = tk.Label(progress, text='A procurar o treino mais recente', bg=CARD, fg=MUTED, anchor='w')
        self.opponent.grid(row=0,column=0,sticky='w')
        self.fraction = tk.Label(progress,text='—',bg=CARD,fg=TEXT,font=('Segoe UI',23,'bold'),anchor='w')
        self.fraction.grid(row=1,column=0,sticky='w',pady=(7,8))
        self.percent = tk.Label(progress,text='',bg=CARD,fg=GREEN,font=('Segoe UI',18,'bold'))
        self.percent.grid(row=1,column=1,sticky='e')
        self.bar = tk.Canvas(progress,height=7,bg='#304132',highlightthickness=0)
        self.bar.grid(row=2,column=0,columnspan=2,sticky='ew')
        self.bar.bind('<Configure>',lambda _: self.draw_progress())
        cards = tk.Frame(root,bg=BG)
        cards.grid(row=2,column=0,sticky='ew',padx=28,pady=14)
        self.values = {}
        for column,(key,title,subtitle) in enumerate([
            ('speed','JOGOS / SEGUNDO','Ritmo dos últimos 5 minutos'),
            ('eta','TEMPO RESTANTE','Estimativa ao ritmo recente'),
            ('turns','TURNOS / JOGO','Média da sessão · 2 meios-lances'),
            ('updates','UPDATES DA REDE','Total acumulado do modelo')]):
            cards.grid_columnconfigure(column,weight=1,uniform='cards')
            frame = tk.Frame(cards,bg=CARD,padx=15,pady=13)
            frame.grid(row=0,column=column,sticky='nsew',padx=(0,12) if column<3 else 0)
            tk.Label(frame,text=title,bg=CARD,fg=MUTED,font=('Segoe UI',9,'bold')).pack(anchor='w')
            value = tk.Label(frame,text='—',bg=CARD,fg=TEXT,font=('Segoe UI',21,'bold'))
            value.pack(anchor='w',pady=(4,2)); self.values[key] = value
            tk.Label(frame,text=subtitle,bg=CARD,fg=MUTED,font=('Segoe UI',8)).pack(anchor='w')
        charts = tk.Frame(root,bg=BG)
        charts.grid(row=3,column=0,sticky='nsew',padx=28,pady=(0,14))
        for column in (0,1): charts.grid_columnconfigure(column,weight=1,uniform='chart')
        charts.grid_rowconfigure(0,weight=1)
        self.speed_chart = Chart(charts,'Ritmo do treino','jogos / s',GREEN)
        self.speed_chart.grid(row=0,column=0,sticky='nsew',padx=(0,7))
        self.survival_chart = Chart(charts,'Sobrevivência','turnos / jogo',BLUE)
        self.survival_chart.grid(row=0,column=1,sticky='nsew',padx=(7,0))
        results = tk.Frame(root,bg=CARD,padx=20,pady=14)
        results.grid(row=4,column=0,sticky='ew',padx=28)
        for i in range(4): results.grid_columnconfigure(i,weight=1)
        for i,(key,title,colour) in enumerate([('wins','VITÓRIAS',GREEN),('draws','EMPATES',GOLD),('losses','DERROTAS',RED),('truncated','LIMITE DE JOGADAS',MUTED)]):
            frame = tk.Frame(results,bg=CARD)
            frame.grid(row=0,column=i,sticky='w')
            tk.Label(frame,text=title,bg=CARD,fg=MUTED,font=('Segoe UI',9,'bold')).pack(anchor='w')
            value = tk.Label(frame,text='—',bg=CARD,fg=colour,font=('Segoe UI',24,'bold'))
            value.pack(anchor='w'); self.values[key] = value
        self.reasons = tk.Label(results,text='Resultados acumulados desta campanha',bg=CARD,fg=MUTED,anchor='w',font=('Segoe UI',9))
        self.reasons.grid(row=1,column=0,columnspan=4,sticky='w',pady=(8,0))
        footer = tk.Frame(root,bg=BG)
        footer.grid(row=5,column=0,sticky='ew',padx=28,pady=(14,20))
        footer.grid_columnconfigure(0,weight=1)
        self.updated = tk.Label(footer,text='Leitura local · atualização a cada 2 segundos',bg=BG,fg=MUTED,anchor='w',font=('Segoe UI',9))
        self.updated.grid(row=0,column=0,sticky='w')
        self.detail = tk.Label(footer,text=str(self.project),bg=BG,fg=MUTED,anchor='w',font=('Segoe UI',8))
        self.detail.grid(row=1,column=0,sticky='w',pady=(4,0))
        for column,(text,command) in enumerate([('Ver logs',self.open_log),('Checkpoints',self.open_checkpoint),('Escolher pasta',self.choose_project)],1):
            tk.Button(footer,text=text,command=command,bg=CARD,fg=TEXT,activebackground='#304132',activeforeground=TEXT,
                      relief='flat',bd=0,padx=12,pady=8,cursor='hand2').grid(row=0,column=column,rowspan=2,padx=(8,0))
        self.refresh()

    def draw_progress(self):
        self.bar.delete('all')
        ratio = min(1, self.snapshot['games']/max(1,self.snapshot['total'])) if self.snapshot else 0
        self.bar.create_rectangle(0,0,self.bar.winfo_width()*ratio,7,fill=GREEN,outline='')

    def refresh(self):
        try:
            self.snapshot = data = self.reader.snapshot()
            if data:
                labels = {'starting':'A iniciar', 'running':'●  Em treino', 'completed':'Concluído', 'failed':'Erro no treino',
                          'stopped':'Parado', 'interrupted':'Treino interrompido', 'stale':'Sem dados recentes'}
                self.status.configure(text=labels[data['state']],fg=GREEN if data['state'] in ('running','completed') else GOLD)
                self.fraction.configure(text=f"{number(data['games'])} / {number(data['total'])} jogos")
                self.percent.configure(text=f"{100*data['games']/max(1,data['total']):.1f}%")
                opponent = data['launch'].get('opponent',{})
                self.opponent.configure(text=f"{opponent.get('engine_name','Adversário externo')}  ·  {opponent.get('elo','—')} Elo nominal  ·  Objetivo: sobrevivência")
                if 'metrics' in data:
                    for key in ('wins','draws','losses','truncated'): self.values[key].configure(text=number(data['counts'][key]))
                    self.values['speed'].configure(text=f"{data['speed']:.2f}")
                    self.values['eta'].configure(text=eta(data['eta']))
                    self.values['turns'].configure(text=f"{data['turns']:.1f}")
                    self.values['updates'].configure(text=number(data['metrics'].get('train_batches',0)))
                    self.speed_chart.render(data['rates']); self.survival_chart.render(data['duration'])
                    names = {'stalemate':'Afogamento', 'threefold_repetition':'Repetição tripla', 'fivefold_repetition':'Repetição quíntupla',
                             'insufficient_material':'Material insuficiente','fifty_moves':'50 lances','seventyfive_moves':'75 lances'}
                    reasons = '   ·   '.join(f'{names.get(k,k)}: {v}' for k,v in data['draw_reasons'].items() if v)
                    self.reasons.configure(text=reasons or 'Resultados acumulados desta campanha · ainda sem empates')
                    updated = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')
                    self.updated.configure(text=f"Últimos dados: {updated}  ·  há {int(data['age'])} s  ·  leitura local a cada 2 s")
                    self.detail.configure(text=f"PID {data['launch'].get('pid','—')}  ·  {data['launch']['run_id']}")
                self.draw_progress()
            else:
                self.status.configure(text='Sem treino encontrado',fg=GOLD)
                self.updated.configure(text='Escolhe a pasta do projeto que contém a pasta history.')
        except Exception as error:
            self.status.configure(text='Leitura temporariamente indisponível',fg=GOLD)
            self.updated.configure(text=str(error)[:130])
        self.root.after(2000,self.refresh)

    def open_log(self):
        if self.snapshot: os.startfile(self.snapshot['launch']['history'])

    def open_checkpoint(self):
        if self.snapshot: os.startfile(str(Path(self.snapshot['launch']['checkpoint']).parent))

    def choose_project(self):
        selected = filedialog.askdirectory(title='Pasta do projeto TonkAI',initialdir=str(self.project))
        if selected:
            self.project = Path(selected)
            self.reader = TrainingReader(selected)
            self.snapshot = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path)
    parser.add_argument('--smoke-test', type=Path)
    args = parser.parse_args()
    base = Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
    project = args.project or (base / 'tracker-config.json')
    if not args.project:
        project = Path(json.loads(project.read_text(encoding='utf-8-sig'))['project']) if project.exists() else base.parent
    if os.name == 'nt':
        ctypes.windll.user32.SetProcessDPIAware()
    root = tk.Tk()
    app = Tracker(root,project)
    if args.smoke_test:
        def capture():
            from PIL import ImageGrab
            root.update()
            box = (root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
            ImageGrab.grab().crop(box).save(args.smoke_test)
            result = {k:app.snapshot[k] for k in ('state','games','total','speed')}
            result['chart_height'] = app.speed_chart.winfo_height()
            args.smoke_test.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            if sys.stdout:
                print(json.dumps(result,ensure_ascii=False))
            root.destroy()
        root.after(2500,capture)
    root.mainloop()


if __name__ == '__main__': main()
