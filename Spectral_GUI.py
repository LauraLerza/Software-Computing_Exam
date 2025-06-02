#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Lerza
"""
import PySimpleGUI as sg
import numpy as np
import sys, os
import plotly.graph_objects as go
from pathlib import Path

# Path to the logo
#It is presumed that the PNG file of the logo is located in the same directory as the script
logo_path = Path(__file__).resolve().parent / "Logo_Laura_resized.png"

# Set the base path for the atomic line files 
base_data_path = Path(__file__).resolve().parent / "Atomic_Lines_NIST"


def make_win2():
    # Left column with buttons
    left_column = [
        [sg.Button('Fix a frame'), sg.Button('Sum all over the counts')],
        [sg.Button('Save calibrated spectrum in a txt file'), sg.Button('Save fit data in a txt file')],
        [sg.Button('Exit')]
    ]

    # Right column with the logo vertically centered
    right_column = [
        [sg.Text('', size=(1, 4))],  # Space above
        [sg.Image(filename=str(logo_path))],
        [sg.Text('', size=(1, 4))]   # Space below
    ]

    # Combined Layout 
    layout = [
        [sg.Column(left_column), sg.VSeparator(), sg.Column(right_column)]
    ]
    
    return sg.Window('Simulated + Experimental spectrum', layout, finalize=True)

# Layout for file selection and mode buttons
file_list_column = [
    [sg.Text("Choose file from folder (for experimental spectrum): "), sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")],
    # Added buttons for selection mode vertical scale
    [sg.Text("Vertical scale mode:"), 
     sg.Radio("Autoscale", "ScaleMode", default=True, key="-MODE_AUTO-"), 
     sg.Radio("Normalize", "ScaleMode", key="-MODE_NORM-"), 
     sg.Radio("Offset-correct", "ScaleMode", key="-MODE_OFFSET-")],
    [sg.Button('Experimental Spectrum'), sg.Button('Simulated Spectrum'), sg.Button('Simulated + Experimental Spectrum')],
    [sg.Button('Exit') ]  # Watermark insertion (logo "Laura Lerza") in the bottom right]
]

# Logo centered vertically simulating margin above and below
logo_column = [
    [sg.Text('', size=(1, 10))],  # space above to center
    [sg.Image(filename=str(logo_path))],
    [sg.Text('', size=(1, 10))]   # space below
]

# Final layout with two columns
layout = [
    [sg.Column(file_list_column), sg.VSeparator(), sg.Column(logo_column, element_justification='right')]
]

# Window
#layout = [[sg.Column(file_list_column)]]
window = sg.Window('Interactive Data Analysis', layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".txt", ".csv"))]
        window["-FILE LIST-"].update(fnames)
    elif event == "Experimental Spectrum":
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            # Portability: use pathlib to parse the file path
            path_parts = Path(filename).parts
            if "Ocean_Optics" in path_parts:
                spectrometer_name = "Ocean_Optics"
            
                
            else:
                spectrometer_name = "Unknown"
            fil = Path(filename).name
            # Optimization: single file loading (avoids double reading)
            data = np.loadtxt(filename, skiprows=1, dtype=np.float32)
            x_array = data[:, 0] + 1.17  # spectral shift (calibration Ocean Optics)
            y_array = data[:, 1:]
            lambda_array = y_array  # alias for compatibility with original code
            shot_num = fil.split('_')[0]

            fig = go.Figure()
            # Adds a trace for each time step (slider)
            for step in range(y_array.shape[1]):
                fig.add_trace(go.Scattergl(visible=False, line=dict(color="#00CED1", width=2),
                                           name=f"time = {step*20} ms", x=x_array, y=y_array[:, step]))
            fig.data[10].visible = True  # for example, view the 10th step initially
            #fig.add_annotation(dict(font=dict(color='darkblue', size=20), x=434.8, y=1180, showarrow=False, text="Ar I", textangle=0, xref="x", yref="y"))
            # Set the Y axis range based on the spectrometer
            if spectrometer_name == "Ocean_Optics":
                fig.update_yaxes(range=[805, np.amax(y_array) + 0.05 * np.amax(y_array)])
            
                
                
                    
            else:
                fig.update_yaxes(range=[1000, np.amax(y_array) + 0.05 * np.amax(y_array)])
            # Slider for time selection
            steps = []
            for i in range(len(fig.data)):
                step = {"method": "update", "args": [{"visible": [False] * len(fig.data)}, {"title": f"Shot: {shot_num}     Spectrum at time: {i*20} ms"}]}
                step["args"][0]["visible"][i] = True
                steps.append(step)
            sliders = [dict(active=10, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)]
            fig.update_layout(sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Counts [a.u.]',
                              font=dict(family="Courier New, monospace", size=20, color="black"))
            fig.show()
            # Selection of the vertical scale mode chosen by the user
            if values.get("-MODE_OFFSET-"):
                # Offset mode: we subtract the minimum value of each frame (baseline) -> baseline of each frame = 0
                y_array = y_array - np.min(y_array, axis=0)
            elif values.get("-MODE_NORM-"):
                # Normalization mode: we divide each frame by its maximum value
                max_vals = np.max(y_array, axis=0)
                max_vals[max_vals == 0] = 1  # let's avoid divisions by 0
                y_array = y_array / max_vals
            # (Autoscale Mode: no changes to the data, autoscaling managed by dynamically setting the Y axis)
            
            fig = go.Figure()
            steps = []
            num_frames = y_array.shape[1]
            for i in range(num_frames):
                fig.add_trace(go.Scattergl(
                    visible=False,
                    line=dict(color="#00CED1", width=2),
                    name=f"time = {i*20} ms",    # track name with time in ms
                    x=x_array,
                    y=y_array[:, i]
                ))
                # Create step slider for the i-th frame
                step = dict(
                    method="update",
                    args=[{"visible": [False]*num_frames},   # all the invisible traces
                          {"title": f"Shot: {shot_num}     Spectrum at time: {i*20} ms"}]  # Updated title
                )
                step["args"][0]["visible"][i] = True        # only the current track is made visible
                # In Autoscale/Normalize mode, it dynamically updates the Y range for this frame.
                if not values.get("-MODE_OFFSET-"):
                    y_min = float(np.min(y_array[:, i]))
                    y_max = float(np.max(y_array[:, i]))
                    if y_max - y_min == 0:
                        y_max += 0.1 * (1 if y_max == 0 else y_max)  # avoid range zero
                    step["args"][1]["yaxis"] = {"range": [y_min, y_max * 1.05]}
                steps.append(step)
            # Set the initial frame visible (e.g. the tenth frame or the last available one)
            initial_idx = 10 if num_frames > 10 else num_frames - 1
            if initial_idx < 0: initial_idx = 0
            fig.data[initial_idx].visible = True
            
            # Set up the slider with the calculated steps
            sliders = [dict(active=initial_idx, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)]
            # If Offset mode, fix the common Y axis from 0 to the global max (baseline aligned to 0)
            if values.get("-MODE_OFFSET-"):
                global_max = float(np.max(y_array))
                fig.update_yaxes(range=[0, global_max * 1.05])
            
            # Update the chart layout by including sliders and axis titles.
            fig.update_layout(
                title={'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
                sliders=sliders,
                xaxis_title='Wavelength [nm]',
                yaxis_title=('Normalized intensity (a.u.)' if values.get("-MODE_NORM-") else 
                             ('Intensity (offset-corrected) [a.u.]' if values.get("-MODE_OFFSET-") else 'Counts [a.u.]')),
                font=dict(family="Courier New, monospace", size=20, color="black")
            )
            fig.show()
            
        except Exception as e:
            print(f"Error displaying experimental spectrum: {e}")
    elif event == "Simulated Spectrum":
        try:
            # Initialize user input for textual mode
            user_inp = ""
            user_stop = "stop"
            # Paths of atomic data files (cross-platform compatible)
            PATH1 = base_data_path / "Argon_I.csv"
            PATH2 = base_data_path / "Argon_II.csv"
            PATH3 = base_data_path / "Fluorine_I.csv"
            PATH4 = base_data_path / "Helium_I.csv"
            PATH5 = base_data_path / "Hydrogen_I.csv"
            PATH6 = base_data_path / "H_F_He.csv"
            # Loading theoretical spectral data (NIST database)
            data1 = np.loadtxt(PATH1, dtype=float)
            data2 = np.loadtxt(PATH2, dtype=float)
            data3 = np.loadtxt(PATH3, dtype=float)
            data4 = np.loadtxt(PATH4, dtype=float)
            data5 = np.loadtxt(PATH5, dtype=float)
            data6 = np.loadtxt(PATH6, dtype=float)
            xcoords1, ycoords1 = data1[:,0], data1[:,1]
            xcoords2, ycoords2 = data2[:,0], data2[:,1]
            xcoords3, ycoords3 = data3[:,0], data3[:,1]
            xcoords4, ycoords4 = data4[:,0], data4[:,1]
            xcoords5, ycoords5 = data5[:,0], data5[:,1]
            xcoords6, ycoords6 = data6[:,0], data6[:,1]
            # Extract names of theoretical datasets (without extension)
            name1 = PATH1.stem
            name2 = PATH2.stem
            name3 = PATH3.stem
            name4 = PATH4.stem
            name5 = PATH5.stem
            name6 = PATH6.stem
            # Low intensity line filter (different thresholds for each element)
            # Optimization: using NumPy masks to filter instead of list comprehension
            if name1 == "Argon_I":
                mask1 = ycoords1 >= 2000
                new_xcoords1 = xcoords1[mask1]
                new_ycoords1 = ycoords1[mask1]
            if name2 == "Argon_II":
                mask2 = ycoords2 >= 1500
                new_xcoords2 = xcoords2[mask2]
                new_ycoords2 = ycoords2[mask2]
            if name3 == "Fluorine_I":
                mask3 = ycoords3 >= 4000
                new_xcoords3 = xcoords3[mask3]
                new_ycoords3 = ycoords3[mask3]
            if name4 == "Helium_I":
                mask4 = ycoords4 >= 200
                new_xcoords4 = xcoords4[mask4]
                new_ycoords4 = ycoords4[mask4]
            if name5 == "Hydrogen_I":
                mask5 = ycoords5 >= 9000
                new_xcoords5 = xcoords5[mask5]
                new_ycoords5 = ycoords5[mask5]
            if name6 == "H_F_He":
                mask6 = ycoords6 >= 1
                new_xcoords6 = xcoords6[mask6]
                new_ycoords6 = ycoords6[mask6]
            # Defines a function for the sum of Gaussians (with vectorization)
            def SumGauss(x, mu, sigma, intensity=1):
                # Optimization: vector calculation of Gaussians for efficiency
                x = np.asarray(x, dtype=float)
                mu = np.asarray(mu, dtype=float)
                intensity = np.asarray(intensity, dtype=float)
                if intensity.ndim == 0:
                    intensity = np.full(mu.shape, intensity)
                sigma_val = float(sigma) if np.isscalar(sigma) or np.ndim(sigma) == 0 else np.asarray(sigma, dtype=float)
                diff = x[:, None] - mu[None, :]
                exponent = -0.5 * ((diff / sigma_val) ** 2)
                gauss_matrix = np.exp(exponent)
                y_vals = gauss_matrix * intensity
                return np.sum(y_vals, axis=1)
            np.seterr(divide='ignore', invalid='ignore')
            # Loop for user input (select dataset to simulate)print("Simulated spectrum mode. Type 'stop' to exit.")
            while user_inp != user_stop:
                user_inp = input("Enter the track number to simulate (trace1=Argon_I, trace2=Argon_II, trace3=Fluorine_I, trace4=Helium_I, trace5=Hydrogen_I, trace6=H+F+He): ")
                if user_inp == "stop":
                    print("Closing simulated spectrum mode.")
                    break
                elif user_inp == "trace1":
                    fig1 = go.Figure()
                    fig1.update_xaxes(range=[new_xcoords1.min()-10, new_xcoords1.max()+10])
                    fig1.update_yaxes(range=[new_ycoords1.min(), new_ycoords1.max() + 0.05*new_ycoords1.max()])
                    # Draw vertical lines for atomic line positions (blue)
                    for k in range(len(new_xcoords1)):
                        fig1.add_shape(type="line", xref='x', yref='y',
                                       x0=new_xcoords1[k], y0=new_ycoords1.min(), x1=new_xcoords1[k], y1=new_ycoords1[k],
                                       line=dict(color="cornflowerblue", width=1))
                    fig1.update_layout(title_text=f"Atomic lines of {name1} in range: {new_xcoords1.min():.2f}-{new_xcoords1.max():.2f} nm",
                                       title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                    x_range1 = np.linspace(new_xcoords1.min()-20, new_xcoords1.max()+20, 60000)
                    # Adds trace for each sigma value (sigma slider)
                    for step in np.arange(0, 1, 0.01):
                        fig1.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                    name=f"σ = {step}", x=x_range1, y=SumGauss(x_range1, new_xcoords1, step, new_ycoords1)))
                    fig1.data[4].visible = True  # show an initial simulated curve (σ ~ 0.03)
                    steps = []
                    for i in range(len(fig1.data)):
                        step = {"method": "update", "args": [{"visible": [False] * len(fig1.data)}]}
                        step["args"][0]["visible"][i] = True
                        steps.append(step)
                    sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                    fig1.update_layout(showlegend=False, sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Intensity [a.u.]', font=dict(family="Courier New, monospace", size=20))
                    fig1.show()
                elif user_inp == "trace2":
                    fig2 = go.Figure()
                    fig2.update_xaxes(range=[new_xcoords2.min()-10, new_xcoords2.max()+10])
                    fig2.update_yaxes(range=[new_ycoords2.min(), new_ycoords2.max() + 0.05*new_ycoords2.max()])
                    for k in range(len(new_xcoords2)):
                        fig2.add_shape(type="line", xref='x', yref='y',
                                       x0=new_xcoords2[k], y0=new_ycoords2.min(), x1=new_xcoords2[k], y1=new_ycoords2[k],
                                       line=dict(color="cornflowerblue", width=1))
                    fig2.update_layout(title_text=f"Atomic lines of {name2} in range: {new_xcoords2.min():.2f}-{new_xcoords2.max():.2f} nm",
                                       title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                    x_range2 = np.linspace(new_xcoords2.min()-20, new_xcoords2.max()+20, 60000)
                    for step in np.arange(0, 1, 0.01):
                        fig2.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                    name=f"σ = {step}", x=x_range2, y=SumGauss(x_range2, new_xcoords2, step, new_ycoords2)))
                    fig2.data[4].visible = True
                    steps = []
                    for i in range(len(fig2.data)):
                        step = {"method": "update", "args": [{"visible": [False] * len(fig2.data)}]}
                        step["args"][0]["visible"][i] = True
                        steps.append(step)
                    sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                    fig2.update_layout(showlegend=False, sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Intensity [a.u.]', font=dict(family="Courier New, monospace", size=20))
                    fig2.show()
                elif user_inp == "trace3":
                    fig3 = go.Figure()
                    fig3.update_xaxes(range=[new_xcoords3.min()-10, new_xcoords3.max()+10])
                    fig3.update_yaxes(range=[new_ycoords3.min(), new_ycoords3.max() + 0.05*new_ycoords3.max()])
                    for k in range(len(new_xcoords3)):
                        fig3.add_shape(type="line", xref='x', yref='y',
                                       x0=new_xcoords3[k], y0=new_ycoords3.min(), x1=new_xcoords3[k], y1=new_ycoords3[k],
                                       line=dict(color="cornflowerblue", width=1))
                    fig3.update_layout(title_text=f"Atomic lines of {name3} in range: {new_xcoords3.min():.2f}-{new_xcoords3.max():.2f} nm",
                                       title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                    x_range3 = np.linspace(new_xcoords3.min()-20, new_xcoords3.max()+20, 60000)
                    for step in np.arange(0, 1, 0.01):
                        fig3.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                    name=f"σ = {step}", x=x_range3, y=SumGauss(x_range3, new_xcoords3, step, new_ycoords3)))
                    fig3.data[4].visible = True
                    steps = []
                    for i in range(len(fig3.data)):
                        step = {"method": "update", "args": [{"visible": [False] * len(fig3.data)}]}
                        step["args"][0]["visible"][i] = True
                        steps.append(step)
                    sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                    fig3.update_layout(showlegend=False, sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Counts [a.u.]', font=dict(family="Courier New, monospace", size=20))
                    fig3.show()
                elif user_inp == "trace4":
                    fig4 = go.Figure()
                    fig4.update_xaxes(range=[new_xcoords4.min()-10, new_xcoords4.max()+10])
                    fig4.update_yaxes(range=[new_ycoords4.min(), new_ycoords4.max() + 0.05*new_ycoords4.max()])
                    for k in range(len(new_xcoords4)):
                        fig4.add_shape(type="line", xref='x', yref='y',
                                       x0=new_xcoords4[k], y0=new_ycoords4.min(), x1=new_xcoords4[k], y1=new_ycoords4[k],
                                       line=dict(color="cornflowerblue", width=1))
                    fig4.update_layout(title_text=f"Atomic lines of {name4} in range: {new_xcoords4.min():.2f}-{new_xcoords4.max():.2f} nm",
                                       title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                    x_range4 = np.linspace(new_xcoords4.min()-20, new_xcoords4.max()+20, 60000)
                    for step in np.arange(0, 1, 0.01):
                        fig4.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                    name=f"σ = {step}", x=x_range4, y=SumGauss(x_range4, new_xcoords4, step, new_ycoords4)))
                    fig4.data[4].visible = True
                    steps = []
                    for i in range(len(fig4.data)):
                        step = {"method": "update", "args": [{"visible": [False] * len(fig4.data)}]}
                        step["args"][0]["visible"][i] = True
                        steps.append(step)
                    sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                    fig4.update_layout(showlegend=False, sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Counts [a.u.]', font=dict(family="Courier New, monospace", size=20))
                    fig4.show()
                elif user_inp == "trace5":
                    fig5 = go.Figure()
                    # Annotations for main lines of emission (H_α, H_β, etc.)
                    fig5.add_annotation(dict(font=dict(color='darkblue', size=20), x=656.27, y=520000, showarrow=False, text="$H_α$", textangle=0, xref="x", yref="y"))
                    fig5.add_annotation(dict(font=dict(color='darkblue', size=20), x=486.13, y=200000, showarrow=False, text="$H_β$", textangle=0, xref="x", yref="y"))
                    fig5.add_annotation(dict(font=dict(color='darkblue', size=20), x=434.04, y=110000, showarrow=False, text="$H_γ$", textangle=0, xref="x", yref="y"))
                    fig5.add_annotation(dict(font=dict(color='darkblue', size=20), x=410.17, y=90000, showarrow=False, text="$H_δ$", textangle=0, xref="x", yref="y"))
                    fig5.add_annotation(dict(font=dict(color='darkblue', size=20), x=397.00, y=50000, showarrow=False, text="$H_ε$", textangle=0, xref="x", yref="y"))
                    fig5.update_xaxes(range=[385, 800])
                    fig5.update_yaxes(range=[0, new_ycoords5.max() + 0.10*new_ycoords5.max()])
                    for k in range(len(new_xcoords5)):
                        fig5.add_shape(type="line", xref='x', yref='y',
                                       x0=new_xcoords5[k], y0=0, x1=new_xcoords5[k], y1=new_ycoords5[k],
                                       line=dict(color="cornflowerblue", width=1))
                    fig5.update_layout(title_text="Simulated spectrum with neutral {} as working gas".format(name5.split("_")[0]),
                                       title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                    #---------Set the range for simulated spectra and points number
                    x_range5 = np.linspace(new_xcoords5.min()-20, 800, 60000)
                    for step in np.arange(0, 1, 0.01):
                        fig5.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                    name=f"σ = {step}", x=x_range5, y=SumGauss(x_range5, new_xcoords5, step, new_ycoords5)))
                    fig5.data[4].visible = True
                    steps = []
                    for i in range(len(fig5.data)):
                        step = {"method": "update", "args": [{"visible": [False] * len(fig5.data)}]}
                        step["args"][0]["visible"][i] = True
                        steps.append(step)
                    sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                    fig5.update_layout(showlegend=False,
                                       title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                       sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Intensity [a.u.]',
                                       font=dict(family="Courier New, monospace", size=20))
                    fig5.show()
                elif user_inp == "trace6":
                    fig6 = go.Figure()
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=20), x=656.27, y=520000, showarrow=False, text="$H_α$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=20), x=486.13, y=200000, showarrow=False, text="$H_β$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=15), x=685.60, y=70000, showarrow=False, text="$F_I$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=20), x=434.04, y=110000, showarrow=False, text="$H_γ$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=20), x=410.17, y=90000, showarrow=False, text="$H_δ$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=20), x=397.00, y=50000, showarrow=False, text="$H_ε$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=15), x=703.70, y=67000, showarrow=False, text="$F_I$", textangle=0, xref="x", yref="y"))
                    fig6.add_annotation(dict(font=dict(color='darkblue', size=15), x=712.80, y=50000, showarrow=False, text="$F_I$", textangle=0, xref="x", yref="y"))
                    fig6.update_xaxes(range=[390, new_xcoords6.max()+10])
                    fig6.update_yaxes(range=[new_ycoords6.min(), new_ycoords6.max() + 0.10*new_ycoords6.max()])
                    for k in range(len(new_xcoords6)):
                        fig6.add_shape(type="line", xref='x', yref='y',
                                       x0=new_xcoords6[k], y0=new_ycoords6.min(), x1=new_xcoords6[k], y1=new_ycoords6[k],
                                       line=dict(color="cornflowerblue", width=1))
                    fig6.update_layout(title_text="Simulated spectrum with H I as working gas, F I as impurity",
                                       title_font_size=20, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                    x_range6 = np.linspace(390, new_xcoords6.max()+10, 60000)
                    for step in np.arange(0, 1, 0.01):
                        fig6.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=3.3),
                                                    name=f"σ = {step}", x=x_range6, y=SumGauss(x_range6, new_xcoords6, step, new_ycoords6)))
                    fig6.data[4].visible = True
                    steps = []
                    for i in range(len(fig6.data)):
                        step = {"method": "update", "args": [{"visible": [False] * len(fig6.data)}]}
                        step["args"][0]["visible"][i] = True
                        steps.append(step)
                    sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                    fig6.update_layout(showlegend=False,
                                       title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                       sliders=sliders, xaxis_title='Wavelength [nm]', yaxis_title='Intensity [a.u.]',
                                       font=dict(family="Courier New, monospace", size=20))
                    fig6.show()
        except Exception as e:
            print(f"Error in spectrum simulation: {e}")
    elif event == "Simulated + Experimental Spectrum":
        try:
            window2 = make_win2()
            window2, event2, values2 = sg.read_all_windows()
            if event2 == sg.WIN_CLOSED or event2 == 'Exit':
                window2.close()
            elif event2 == 'Fix a frame':
                # --- Reading of the selected display mode (autoscale, normalize, offset)
                mode_auto   = values2.get("-MODE_AUTO-")   # Independent vertical scale mode (separate Y axis for simulated)
                mode_norm   = values2.get("-MODE_NORM-")   # Common scale mode with normalization
                mode_offset = values2.get("-MODE_OFFSET-") # Common scale mode with offset correction

                # --- Set the selected experimental file and upload the datati
                from pathlib import Path
                file_path = Path(values["-FOLDER-"]) / values["-FILE LIST-"][0]   # Complete path of the selected file
                filename = str(file_path)
                # Loading array of wavelengths and intensities from the file (experimental spectrum as a function of time)
                wavelength_array = np.loadtxt(filename, skiprows=1, dtype=np.float32)
                lambda_array = np.loadtxt(filename, skiprows=1, dtype=np.float32)[:, 1:]
                # Detect the spectrometer from the path and apply the appropriate offset in wavelength.
                if "Ocean_Optics" in filename:
                    x_array = wavelength_array[:, 0] + 1.17    # shift lambda for Ocean Optics
                
                else:
                    x_array = wavelength_array[:, 0]
                y_array = lambda_array[:, :]

                # --- Determine the time frame with maximum intensity and calculate the baseline and maximum value.
                result = np.where(lambda_array == np.amax(lambda_array))
                frame_idx = int(result[1][0])                           # index of the frame (column) with maximum intensity
                baseline_exp = float(np.amin(y_array[:, frame_idx]))    # minimum value (baseline) of the selected frame
                max_exp = float(np.amax(y_array[:, frame_idx]))         # maximum intensity of the selected frame

                # --- Control variables for user input loop
                user_inp = 0
                user_stop = 'stop'
                
                # initial value of the slider (σ ≈ 0.04)
                #initial_step = 4  
                # --- Paths of the atomic lines files (using pathlib for macOS/Windows compatibility)
                # base_dir = Path(r'C:\Users\lau65\OneDrive\Desktop\Codice_Laura\Atomic_Lines_NIST')
                

                PATH1 = str(Path(__file__).resolve().parent / "Atomic_Lines_NIST" / "Argon_I.csv")
                PATH2 = str(Path(__file__).resolve().parent / "Atomic_Lines_NIST" / "Argon_II.csv")
                PATH3 = str(Path(__file__).resolve().parent / "Atomic_Lines_NIST" / "Fluorine_I.csv")
                PATH4 = str(Path(__file__).resolve().parent / "Atomic_Lines_NIST" / "Helium_I.csv")
                PATH5 = str(Path(__file__).resolve().parent / "Atomic_Lines_NIST" / "Hydrogen_I_normalized.csv")
                PATH6 = str(Path(__file__).resolve().parent / "Atomic_Lines_NIST" / "H_F_He.csv")

                # --- Loading theoretical data for atomic lines from CSV file
                data1 = np.loadtxt(PATH1)
                xcoords1 = data1[:, 0]; ycoords1 = data1[:, 1]
                data2 = np.loadtxt(PATH2)
                xcoords2 = data2[:, 0]; ycoords2 = data2[:, 1]   # Correction: use data2 for Argon_II
                data3 = np.loadtxt(PATH3)
                xcoords3 = data3[:, 0]; ycoords3 = data3[:, 1]
                data4 = np.loadtxt(PATH4)
                xcoords4 = data4[:, 0]; ycoords4 = data4[:, 1]
                data5 = np.loadtxt(PATH5)
                xcoords5 = data5[:, 0]; ycoords5 = data5[:, 1]
                data6 = np.loadtxt(PATH6)
                xcoords6 = data6[:, 0]; ycoords6 = data6[:, 1]

                # --- Names of atomic series (from the file name, without extension)
                name1 = Path(PATH1).stem
                name2 = Path(PATH2).stem
                name3 = Path(PATH3).stem
                name4 = Path(PATH4).stem
                name5 = Path(PATH5).stem
                name6 = Path(PATH6).stem

                # --- Filter the weak atomic lines below a threshold (for each theoretical dataset)
                if name1 == 'Argon_I':
                    new_ycoords1 = [i for i in ycoords1 if i >= 2000]
                    new_xcoords1 = [row[0] for row in data1 if row[1] >= 2000]
                else:
                    new_xcoords1, new_ycoords1 = xcoords1, ycoords1
                xmin1 = np.amin(new_xcoords1); xmax1 = np.amax(new_xcoords1)
                ymin1 = np.amin(new_ycoords1); ymax1 = np.amax(new_ycoords1)

                if name2 == 'Argon_II':
                    new_ycoords2 = [i for i in ycoords2 if i >= 1500]
                    new_xcoords2 = [row[0] for row in data2 if row[1] >= 1500]
                else:
                    new_xcoords2, new_ycoords2 = xcoords2, ycoords2
                xmin2 = np.amin(new_xcoords2); xmax2 = np.amax(new_xcoords2)
                ymin2 = np.amin(new_ycoords2); ymax2 = np.amax(new_ycoords2)

                if name3 == 'Fluorine_I':
                    new_ycoords3 = [i for i in ycoords3 if i >= 4000]
                    new_xcoords3 = [row[0] for row in data3 if row[1] >= 4000]
                else:
                    new_xcoords3, new_ycoords3 = xcoords3, ycoords3
                xmin3 = np.amin(new_xcoords3); xmax3 = np.amax(new_xcoords3)
                ymin3 = np.amin(new_ycoords3); ymax3 = np.amax(new_ycoords3)

                if name4 == 'Helium_I':
                    new_ycoords4 = [i for i in ycoords4 if i >= 200]
                    new_xcoords4 = [row[0] for row in data4 if row[1] >= 200]
                else:
                    new_xcoords4, new_ycoords4 = xcoords4, ycoords4
                xmin4 = np.amin(new_xcoords4); xmax4 = np.amax(new_xcoords4)
                ymin4 = np.amin(new_ycoords4); ymax4 = np.amax(new_ycoords4)

                if name5 == 'Hydrogen_I_normalized':
                    # 9000/31 ≈ 290.3 to normalize the more intense peak at ~9000 (shot 1228)
                    new_ycoords5 = [i for i in ycoords5 if i >= 290.3]
                    new_xcoords5 = [row[0] for row in data5 if row[1] >= 290.3]
                else:
                    new_xcoords5, new_ycoords5 = xcoords5, ycoords5
                xmin5 = np.amin(new_xcoords5); xmax5 = np.amax(new_xcoords5)
                ymin5 = np.amin(new_ycoords5); ymax5 = np.amax(new_ycoords5)

                if name6 == 'Hydrogen_I_normalized_1022':
                    # 9000/230 ≈ 39 to normalize the more intense peak at ~9000 (shot 1022)
                    new_ycoords6 = [i for i in ycoords6 if i >= 39]
                    new_xcoords6 = [row[0] for row in data6 if row[1] >= 39]
                else:
                    new_xcoords6, new_ycoords6 = xcoords6, ycoords6
                xmin6 = np.amin(new_xcoords6); xmax6 = np.amax(new_xcoords6)
                ymin6 = np.amin(new_ycoords6); ymax6 = np.amax(new_ycoords6)

                # --- Prepare normalized intensities for the simulations (max intensity = 1) - used in Normalize mode.
                norm_new_ycoords1 = new_ycoords1 / np.amax(new_ycoords1) if len(new_ycoords1)>0 else new_ycoords1
                norm_new_ycoords2 = new_ycoords2 / np.amax(new_ycoords2) if len(new_ycoords2)>0 else new_ycoords2
                norm_new_ycoords3 = new_ycoords3 / np.amax(new_ycoords3) if len(new_ycoords3)>0 else new_ycoords3
                norm_new_ycoords4 = new_ycoords4 / np.amax(new_ycoords4) if len(new_ycoords4)>0 else new_ycoords4
                norm_new_ycoords5 = new_ycoords5 / np.amax(new_ycoords5) if len(new_ycoords5)>0 else new_ycoords5
                norm_new_ycoords6 = new_ycoords6 / np.amax(new_ycoords6) if len(new_ycoords6)>0 else new_ycoords6

                # Convolution function (sum of Gaussians) to generate simulated spectra
                def SumGauss(x, mu, sigma, intensity=1):
                    sumgaussian = sum([intensity[i] * np.exp(-0.5 * ((x - mu[i]) / sigma) ** 2)
                                       for i in range(len(mu))])
                    return sumgaussian

                np.seterr(divide='ignore', invalid='ignore')  # ignore divide-by-zero warning for sigma=0

                # --- User input loop to choose the simulated dataset (trace1..trace6)
                while user_inp != user_stop:
                    print("To simulate the spectrum, insert the trace number:")
                    print("trace1=Argon_I\ntrace2=Argon_II\ntrace3=Fluorine_I\ntrace4=Helium_I\ntrace5=Hydrogen_I_normalized\ntrace6=H_impurities_F_He")
                    print("N.B.: to close the program, digit the string 'stop'.")
                    user_inp = input()
                    if user_inp == "stop":
                        print("Program properly closed.")
                        sys.exit()

                    elif user_inp == "trace1":
                        # --- Definition of Plotly figure for Argon_I
                        fig1 = go.Figure()
                        # --- Data for slider σ (array placeholder of length 100)
                        sigma_values = np.arange(0, 1, 0.01)  # 0.00, 0.01, ..., 0.99 (100 values)
                        # --- Adds the track of the fixed experimental frame (always index 0)
                        y_exp_plot = y_array[:, frame_idx].copy()
                        if mode_offset:
                            y_exp_plot -= baseline_exp  # subtract offset baseline
                        if mode_norm:
                            y_exp_plot /= max_exp      # normalize experimental intensity
                        fig1.add_trace(go.Scattergl(
                            showlegend=False, visible=True,
                            line=dict(color="mediumseagreen", width=1.5),
                            name=f"Frame at time = {frame_idx} ms",
                            x=x_array, y=y_exp_plot
                        ))
                        # --- Y axis settings based on the mode
                        if mode_auto:
                            # Independent vertical scale: y2 axis for simulated data (autoscale to the right)
                            yaxis_title = "Counts [a.u.]"
                            yaxis2_title = "Relative intensity"
                            fig1.update_layout(
                                yaxis=dict(title=yaxis_title),
                                yaxis2=dict(title=yaxis2_title, overlaying='y', side='right', rangemode='tozero')
                            )
                        else:
                            # Common scale (Normalize/Offset): a single Y axis for both
                            yaxis_title = "Intensity [a.u.]" if mode_offset else "Relative intensity [a.u.]"
                            fig1.update_layout(yaxis=dict(title=yaxis_title))
                        # --- Adds vertical lines for the atomic rows (in blue) on the graph.
                        #     Use yref='y2' for autoscale mode (lines on the secondary axis), otherwise 'y'
                        shape_yref = 'y2' if mode_auto else 'y'
                        # Starting baseline for the shapes (0 if normalized baseline or separate axis)
                        shape_y0 = 0
                        if mode_norm and not mode_auto:
                            shape_y0 = baseline_exp / max_exp  # in Normalize mode, start shape at the minimum normalized intensity
                        elif mode_offset and not mode_auto:
                            shape_y0 = 0  # the experimental baseline has been brought to 0
                        # Choose the array of atomic intensities to use (normalized or original)
                        intensities1 = norm_new_ycoords1 if mode_norm else new_ycoords1
                        for k in range(len(new_xcoords1)):
                            fig1.add_shape(type="line",
                                           xref='x', yref=shape_yref,
                                           x0=new_xcoords1[k], x1=new_xcoords1[k],
                                           y0=shape_y0, y1=float(intensities1[k]),
                                           line=dict(color="cornflowerblue", width=1))
                        # --- Combined chart title (atomic series + experimental frame)
                        fig1.update_layout(title_text=f"Atomic lines of {name1} in range: {xmin1:.2f}-{xmax1:.2f} nm<br>and fixed frame of experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]")
                        # --- Prepare the abscissa for simulated spectra (extended range slightly beyond xmin-xmax)     
                        
                        x_range1 = np.linspace(xmin1 - 20, xmax1 + 20, 60000)
                        # --- Adds simulated tracks (one for each value of σ on the slider)
                        intensities1 = norm_new_ycoords1 if mode_norm else new_ycoords1  # intensity to use (normalized if required)
                        for sigma in sigma_values:
                            fig1.add_trace(go.Scatter(
                                visible=False,
                                line=dict(color="red", width=1.3),
                                name=f"σ = {sigma:.2f}",
                                x=x_range1,
                                y=SumGauss(x_range1, new_xcoords1, sigma, intensities1),
                                yaxis='y2' if mode_auto else 'y'
                            ))
                        # --- Keeps the experimental trace visible (index 0) and sets the initial visibility for simulation (σ ~ 0.04)
                        initial_step = 4  # initial slider index (σ ~ 0.04)
                        if len(fig1.data) > 1:
                            fig1.data[initial_step + 1].visible = True
                        # --- Create the slider for σ: at each step, make only the corresponding simulated trace visible (plus the experimental one)
                        steps = []
                        total_traces = 1 + len(sigma_values)  # total track number (1 exp + N sim)
                        for i, sigma in enumerate(sigma_values):
                            vis = [False] * total_traces
                            vis[0] = True               # the experimental track remains always visible
                            vis[i + 1] = True           # activate the i-th simulated track
                            step = dict(method="update", args=[{"visible": vis}])
                            steps.append(step)
                        sliders = [dict(active=initial_step,
                                        currentvalue={"prefix": "σ: "},
                                        pad={"t": 50},
                                        steps=steps)]
                        fig1.update_layout(showlegend=False, sliders=sliders)
                        fig1.show()

                    elif user_inp == "trace2":
                        # --- Definition figure for Argon_II
                        fig2 = go.Figure()
                        sigma_values = np.arange(0, 1, 0.01)
                        # Fixed experimental track
                        y_exp_plot = y_array[:, frame_idx].copy()
                        if mode_offset:
                            y_exp_plot -= baseline_exp
                        if mode_norm:
                            y_exp_plot /= max_exp
                        fig2.add_trace(go.Scattergl(
                            showlegend=False, visible=True,
                            line=dict(color="mediumseagreen", width=1.5),
                            name=f"Frame at time = {frame_idx} ms",
                            x=x_array, y=y_exp_plot
                        ))
                        # Update the Y axes for selected mode
                        if mode_auto:
                            fig2.update_layout(
                                yaxis=dict(title="Counts [a.u.]"),
                                yaxis2=dict(title="Relative intensity", overlaying='y', side='right', rangemode='tozero')
                            )
                        else:
                            yaxis_title = "Intensity [a.u.]" if mode_offset else "Relative intensity [a.u.]"
                            fig2.update_layout(yaxis=dict(title=yaxis_title))
                        # Atomic line spectra (blue)
                        shape_yref = 'y2' if mode_auto else 'y'
                        shape_y0 = 0
                        if mode_norm and not mode_auto:
                            shape_y0 = baseline_exp / max_exp
                        intensities2 = norm_new_ycoords2 if mode_norm else new_ycoords2
                        for k in range(len(new_xcoords2)):
                            fig2.add_shape(type="line",
                                           xref='x', yref=shape_yref,
                                           x0=new_xcoords2[k], x1=new_xcoords2[k],
                                           y0=shape_y0, y1=float(intensities2[k]),
                                           line=dict(color="cornflowerblue", width=1))
                        fig2.update_layout(title_text=f"Atomic lines of {name2} in range: {xmin2:.2f}-{xmax2:.2f} nm<br>and fixed frame of experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]")
                        x_range2 = np.linspace(xmin2 - 20, xmax2 + 20, 60000)
                        intensities2 = norm_new_ycoords2 if mode_norm else new_ycoords2
                        for sigma in sigma_values:
                            fig2.add_trace(go.Scatter(
                                visible=False,
                                line=dict(color="red", width=1.3),
                                name=f"σ = {sigma:.2f}",
                                x=x_range2,
                                y=SumGauss(x_range2, new_xcoords2, sigma, intensities2),
                                yaxis='y2' if mode_auto else 'y'
                            ))
                        # --- Keeps the experimental trace visible (index 0) and sets initial visibility for simulation. (σ ~ 0.04)
                        initial_step = 4  # initial slider index (σ ~ 0.04)
                        if len(fig2.data) > 1:
                            fig2.data[initial_step + 1].visible = True
                        steps = []
                        total_traces = 1 + len(sigma_values)
                        for i, sigma in enumerate(sigma_values):
                            vis = [False] * total_traces
                            vis[0] = True
                            vis[i + 1] = True
                            steps.append(dict(method="update", args=[{"visible": vis}]))
                        sliders = [dict(active=initial_step,
                                        currentvalue={"prefix": "σ: "},
                                        pad={"t": 50},
                                        steps=steps)]
                        fig2.update_layout(showlegend=False, sliders=sliders)
                        fig2.show()

                    elif user_inp == "trace3":
                        fig3 = go.Figure()
                        sigma_values = np.arange(0, 1, 0.01)
                        y_exp_plot = y_array[:, frame_idx].copy()
                        if mode_offset:
                            y_exp_plot -= baseline_exp
                        if mode_norm:
                            y_exp_plot /= max_exp
                        fig3.add_trace(go.Scattergl(
                            showlegend=False, visible=True,
                            line=dict(color="mediumseagreen", width=1.5),
                            name=f"Frame at time = {frame_idx} ms",
                            x=x_array, y=y_exp_plot
                        ))
                        if mode_auto:
                            fig3.update_layout(
                                yaxis=dict(title="Counts [a.u.]"),
                                yaxis2=dict(title="Relative intensity", overlaying='y', side='right', rangemode='tozero')
                            )
                        else:
                            yaxis_title = "Intensity [a.u.]" if mode_offset else "Relative intensity [a.u.]"
                            fig3.update_layout(yaxis=dict(title=yaxis_title))
                        shape_yref = 'y2' if mode_auto else 'y'
                        shape_y0 = 0
                        if mode_norm and not mode_auto:
                            shape_y0 = baseline_exp / max_exp
                        intensities3 = norm_new_ycoords3 if mode_norm else new_ycoords3
                        for k in range(len(new_xcoords3)):
                            fig3.add_shape(type="line",
                                           xref='x', yref=shape_yref,
                                           x0=new_xcoords3[k], x1=new_xcoords3[k],
                                           y0=shape_y0, y1=float(intensities3[k]),
                                           line=dict(color="cornflowerblue", width=1))
                        fig3.update_layout(title_text=f"Atomic lines of {name3} in range: {xmin3:.2f}-{xmax3:.2f} nm<br>and fixed frame of experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]")
                        x_range3 = np.linspace(xmin3 - 20, xmax3 + 20, 60000)
                        intensities3 = norm_new_ycoords3 if mode_norm else new_ycoords3
                        for sigma in sigma_values:
                            fig3.add_trace(go.Scatter(
                                visible=False,
                                line=dict(color="red", width=1.3),
                                name=f"σ = {sigma:.2f}",
                                x=x_range3,
                                y=SumGauss(x_range3, new_xcoords3, sigma, intensities3),
                                yaxis='y2' if mode_auto else 'y'
                            ))
                        # --- Keeps the experimental trace visible (index 0) and sets initial visibility for simulation (σ ~ 0.04)
                        initial_step = 4  # initial slider index (σ ~ 0.04)
                        if len(fig3.data) > 1:
                            fig3.data[initial_step + 1].visible = True
                        steps = []
                        total_traces = 1 + len(sigma_values)
                        for i, sigma in enumerate(sigma_values):
                            vis = [False] * total_traces
                            vis[0] = True
                            vis[i + 1] = True
                            steps.append(dict(method="update", args=[{"visible": vis}]))
                        sliders = [dict(active=initial_step,
                                        currentvalue={"prefix": "σ: "},
                                        pad={"t": 50},
                                        steps=steps)]
                        fig3.update_layout(showlegend=False, sliders=sliders)
                        fig3.show()

                    elif user_inp == "trace4":
                        fig4 = go.Figure()
                        sigma_values = np.arange(0, 1, 0.01)
                        y_exp_plot = y_array[:, frame_idx].copy()
                        if mode_offset:
                            y_exp_plot -= baseline_exp
                        if mode_norm:
                            y_exp_plot /= max_exp
                        fig4.add_trace(go.Scattergl(
                            showlegend=False, visible=True,
                            line=dict(color="mediumseagreen", width=1.5),
                            name=f"Frame at time = {frame_idx} ms",
                            x=x_array, y=y_exp_plot
                        ))
                        if mode_auto:
                            fig4.update_layout(
                                yaxis=dict(title="Counts [a.u.]"),
                                yaxis2=dict(title="Relative intensity", overlaying='y', side='right', rangemode='tozero')
                            )
                        else:
                            yaxis_title = "Intensity [a.u.]" if mode_offset else "Relative intensity [a.u.]"
                            fig4.update_layout(yaxis=dict(title=yaxis_title))
                        shape_yref = 'y2' if mode_auto else 'y'
                        shape_y0 = 0
                        if mode_norm and not mode_auto:
                            shape_y0 = baseline_exp / max_exp
                        intensities4 = norm_new_ycoords4 if mode_norm else new_ycoords4
                        for k in range(len(new_xcoords4)):
                            fig4.add_shape(type="line",
                                           xref='x', yref=shape_yref,
                                           x0=new_xcoords4[k], x1=new_xcoords4[k],
                                           y0=shape_y0, y1=float(intensities4[k]),
                                           line=dict(color="cornflowerblue", width=1))
                        fig4.update_layout(title_text=f"Atomic lines of {name4} in range: {xmin4:.2f}-{xmax4:.2f} nm<br>and fixed frame of experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]")
                        x_range4 = np.linspace(xmin4 - 20, xmax4 + 20, 60000)
                        intensities4 = norm_new_ycoords4 if mode_norm else new_ycoords4
                        for sigma in sigma_values:
                            fig4.add_trace(go.Scatter(
                                visible=False,
                                line=dict(color="red", width=1.3),
                                name=f"σ = {sigma:.2f}",
                                x=x_range4,
                                y=SumGauss(x_range4, new_xcoords4, sigma, intensities4),
                                yaxis='y2' if mode_auto else 'y'
                            ))
                        # --- Keeps the experimental trace visible (index 0) and sets initial visibility for simulation (σ ~ 0.04)
                        initial_step = 4  # initial slider index (σ ~ 0.04)
                        if len(fig4.data) > 1:
                            fig4.data[initial_step + 1].visible = True
                        steps = []
                        total_traces = 1 + len(sigma_values)
                        for i, sigma in enumerate(sigma_values):
                            vis = [False] * total_traces
                            vis[0] = True
                            vis[i + 1] = True
                            steps.append(dict(method="update", args=[{"visible": vis}]))
                        sliders = [dict(active=initial_step,
                                        currentvalue={"prefix": "σ: "},
                                        pad={"t": 50},
                                        steps=steps)]
                        fig4.update_layout(showlegend=False, sliders=sliders)
                        fig4.show()

                    elif user_inp == "trace5":
                        fig5 = go.Figure()
                        # Adds annotations on Hα and Hβ in the graph (blue text at specific wavelengths)
                        fig5.add_annotation(dict(font=dict(color='darkblue', size=20),
                                                 x=656.27, y=17000, showarrow=False, text="$H_α$", xref="x", yref="y"))
                        fig5.add_annotation(dict(font=dict(color='darkblue', size=20),
                                                 x=486.13, y=5000, showarrow=False, text="$H_β$", xref="x", yref="y"))
                        sigma_values = np.arange(0, 1, 0.01)
                        y_exp_plot = y_array[:, frame_idx].copy()
                        if mode_offset:
                            y_exp_plot -= baseline_exp
                        if mode_norm:
                            y_exp_plot /= max_exp
                        fig5.add_trace(go.Scattergl(
                            showlegend=False, visible=True,
                            line=dict(color="mediumseagreen", width=1.5),
                            name=f"Frame at time = {frame_idx} ms",
                            x=x_array, y=y_exp_plot
                        ))
                        if mode_auto:
                            fig5.update_layout(
                                yaxis=dict(title="Counts [a.u.]"),
                                yaxis2=dict(title="Relative intensity", overlaying='y', side='right', rangemode='tozero')
                            )
                        else:
                            yaxis_title = "Intensity [a.u.]" if mode_offset else "Relative intensity [a.u.]"
                            fig5.update_layout(yaxis=dict(title=yaxis_title))
                        shape_yref = 'y2' if mode_auto else 'y'
                        shape_y0 = 0
                        if mode_norm and not mode_auto:
                            shape_y0 = baseline_exp / max_exp
                        intensities5 = norm_new_ycoords5 if mode_norm else new_ycoords5
                        for k in range(len(new_xcoords5)):
                            fig5.add_shape(type="line",
                                           xref='x', yref=shape_yref,
                                           x0=new_xcoords5[k], x1=new_xcoords5[k],
                                           y0=shape_y0, y1=float(intensities5[k]),
                                           line=dict(color="cornflowerblue", width=1))
                        # Set combined title (for shot 1228 they used a specific title)
                        fig5.update_layout(title_text="Experimental and simulated spectrum of plasma shot 1228",
                                           title_font_size=30, xaxis_title="Wavelength [nm]")
                        x_range5 = np.linspace(xmin5 - 20, xmax5 + 20, 60000)
                        intensities5 = norm_new_ycoords5 if mode_norm else new_ycoords5
                        for sigma in sigma_values:
                            fig5.add_trace(go.Scatter(
                                visible=False,
                                line=dict(color="red", width=1.3),
                                name=f"σ = {sigma:.2f}",
                                x=x_range5,
                                y=SumGauss(x_range5, new_xcoords5, sigma, intensities5),
                                yaxis='y2' if mode_auto else 'y'
                            ))
                        # --- Keeps the experimental trace visible (index 0) and sets initial visibility for simulation (σ ~ 0.04)
                        initial_step = 4  # initial slider index (σ ~ 0.04)
                        if len(fig5.data) > 1:
                            fig5.data[initial_step + 1].visible = True
                        steps = []
                        total_traces = 1 + len(sigma_values)
                        for i, sigma in enumerate(sigma_values):
                            vis = [False] * total_traces
                            vis[0] = True
                            vis[i + 1] = True
                            steps.append(dict(method="update", args=[{"visible": vis}]))
                        sliders = [dict(active=initial_step,
                                        currentvalue={"prefix": "σ: "},
                                        pad={"t": 50},
                                        steps=steps)]
                        fig5.update_layout(showlegend=False,
                                           sliders=sliders,
                                           font=dict(family="Courier New, monospace", size=20, color="black"))
                        fig5.show()

                    elif user_inp == "trace6":
                        fig6 = go.Figure()
                        sigma_values = np.arange(0, 1, 0.01)
                        y_exp_plot = y_array[:, frame_idx].copy()
                        if mode_offset:
                            y_exp_plot -= baseline_exp
                        if mode_norm:
                            y_exp_plot /= max_exp
                        fig6.add_trace(go.Scattergl(
                            showlegend=False, visible=True,
                            line=dict(color="mediumseagreen", width=1.5),
                            name=f"Frame at time = {frame_idx} ms",
                            x=x_array, y=y_exp_plot
                        ))
                        if mode_auto:
                            fig6.update_layout(
                                yaxis=dict(title="Counts [a.u.]"),
                                yaxis2=dict(title="Relative intensity", overlaying='y', side='right', rangemode='tozero')
                            )
                        else:
                            yaxis_title = "Intensity [a.u.]" if mode_offset else "Relative intensity [a.u.]"
                            fig6.update_layout(yaxis=dict(title=yaxis_title))
                        shape_yref = 'y2' if mode_auto else 'y'
                        shape_y0 = 0
                        if mode_norm and not mode_auto:
                            shape_y0 = baseline_exp / max_exp
                        intensities6 = norm_new_ycoords6 if mode_norm else new_ycoords6
                        for k in range(len(new_xcoords6)):
                            fig6.add_shape(type="line",
                                           xref='x', yref=shape_yref,
                                           x0=new_xcoords6[k], x1=new_xcoords6[k],
                                           y0=shape_y0, y1=float(intensities6[k]),
                                           line=dict(color="cornflowerblue", width=1))
                        fig6.update_layout(title_text="Experimental and simulated spectrum of plasma shot 1022",
                                           title_font_size=30, xaxis_title="Wavelength [nm]")
                        x_range6 = np.linspace(xmin6 - 20, xmax6 + 20, 60000)
                        intensities6 = norm_new_ycoords6 if mode_norm else new_ycoords6
                        for sigma in sigma_values:
                            fig6.add_trace(go.Scatter(
                                visible=False,
                                line=dict(color="red", width=1.3),
                                name=f"σ = {sigma:.2f}",
                                x=x_range6,
                                y=SumGauss(x_range6, new_xcoords6, sigma, intensities6),
                                yaxis='y2' if mode_auto else 'y'
                            ))
                        # --- Keep the experimental trace visible (index 0) and set initial visibility for simulation (σ ~ 0.04)
                        initial_step = 4  # initial slider index (σ ~ 0.04)
                        if len(fig6.data) > 1:
                            fig6.data[initial_step + 1].visible = True
                        steps = []
                        total_traces = 1 + len(sigma_values)
                        for i, sigma in enumerate(sigma_values):
                            vis = [False] * total_traces
                            vis[0] = True
                            vis[i + 1] = True
                            steps.append(dict(method="update", args=[{"visible": vis}]))
                        sliders = [dict(active=initial_step,
                                        currentvalue={"prefix": "σ: "},
                                        pad={"t": 50},
                                        steps=steps)]
                        fig6.update_layout(showlegend=False,
                                           sliders=sliders,
                                           font=dict(family="Courier New, monospace", size=20, color="black"))
                        fig6.show()
            elif event2 == 'Sum all over the counts':
                filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
                data = np.loadtxt(filename, skiprows=1, dtype=np.float32)
                x_array = data[:, 0] + 1.22
                y_array = data[:, 1:]
                sum_array = np.sum(y_array, axis=1)
                unique_vals = np.unique(sum_array)
                y_min_val = unique_vals[2] if unique_vals.size > 2 else unique_vals.min()
                PATH1 = base_data_path / "Argon_I.csv"
                PATH2 = base_data_path / "Argon_II.csv"
                PATH3 = base_data_path / "Fluorine_I.csv"
                PATH4 = base_data_path / "Helium_I.csv"
                PATH5 = base_data_path / "Hydrogen_I.csv"
                data1 = np.loadtxt(PATH1, dtype=float)
                data2 = np.loadtxt(PATH2, dtype=float)
                data3 = np.loadtxt(PATH3, dtype=float)
                data4 = np.loadtxt(PATH4, dtype=float)
                data5 = np.loadtxt(PATH5, dtype=float)
                xcoords1, ycoords1 = data1[:,0], data1[:,1]
                xcoords2, ycoords2 = data2[:,0], data2[:,1]
                xcoords3, ycoords3 = data3[:,0], data3[:,1]
                xcoords4, ycoords4 = data4[:,0], data4[:,1]
                xcoords5, ycoords5 = data5[:,0], data5[:,1]
                name1 = PATH1.stem
                name2 = PATH2.stem
                name3 = PATH3.stem
                name4 = PATH4.stem
                name5 = PATH5.stem
                if name1 == "Argon_I":
                    mask1 = ycoords1 >= 2000
                    new_xcoords1 = xcoords1[mask1]; new_ycoords1 = ycoords1[mask1] * 300
                if name2 == "Argon_II":
                    mask2 = ycoords2 >= 1500
                    new_xcoords2 = xcoords2[mask2]; new_ycoords2 = ycoords2[mask2]
                if name3 == "Fluorine_I":
                    mask3 = ycoords3 >= 4000
                    new_xcoords3 = xcoords3[mask3]; new_ycoords3 = ycoords3[mask3]
                if name4 == "Helium_I":
                    mask4 = ycoords4 >= 200
                    new_xcoords4 = xcoords4[mask4]; new_ycoords4 = ycoords4[mask4]
                if name5 == "Hydrogen_I":
                    mask5 = ycoords5 >= 9000
                    new_xcoords5 = xcoords5[mask5]; new_ycoords5 = ycoords5[mask5]
                def SumGauss(x, mu, sigma, intensity=1):
                    x = np.asarray(x, dtype=float)
                    mu = np.asarray(mu, dtype=float)
                    intensity = np.asarray(intensity, dtype=float)
                    if intensity.ndim == 0:
                        intensity = np.full(mu.shape, intensity)
                    sigma_val = float(sigma) if np.isscalar(sigma) or np.ndim(sigma) == 0 else np.asarray(sigma, dtype=float)
                    diff = x[:, None] - mu[None, :]
                    exponent = -0.5 * ((diff / sigma_val) ** 2)
                    gauss_matrix = np.exp(exponent)
                    y_vals = gauss_matrix * intensity
                    return np.sum(y_vals, axis=1)
                np.seterr(divide='ignore', invalid='ignore')
                print("Modalità spettro combinato (somma conteggi). Digitare 'stop' per uscire.")
                user_inp = ""
                while user_inp != "stop":
                    user_inp = input("Inserire il numero della traccia da simulare (trace1=Argon_I, trace2=Argon_II, trace3=Fluorine_I, trace4=Helium_I, trace5=Hydrogen_I): ")
                    if user_inp == "stop":
                        print("Chiusura modalità combinata (somma conteggi).")
                        break
                    elif user_inp == "trace1":
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scattergl(showlegend=False, visible=True, line=dict(color="mediumseagreen", width=1.5),
                                                    x=x_array, y=sum_array))
                        fig1.update_xaxes(range=[x_array.min()-10, x_array.max()+10])
                        fig1.update_yaxes(range=[y_min_val, sum_array.max() + 0.05*sum_array.max()])
                        for k in range(len(new_xcoords1)):
                            fig1.add_shape(type="line", xref='x', yref='y',
                                           x0=new_xcoords1[k], y0=810, x1=new_xcoords1[k], y1=new_ycoords1[k],
                                           line=dict(color="cornflowerblue", width=1))
                        fig1.update_layout(title_text=f"Atomic lines of {name1} in range: {new_xcoords1.min():.2f}-{new_xcoords1.max():.2f} nm<br>and summed counts of the experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                        x_range1 = np.linspace(new_xcoords1.min()-20, new_xcoords1.max()+20, 60000)
                        for step in np.arange(0, 1, 0.01):
                            fig1.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                        name=f"σ = {step}", x=x_range1, y=SumGauss(x_range1, new_xcoords1, step, new_ycoords1)))
                        fig1.data[4].visible = True
                        steps = []
                        total = len(fig1.data)
                        for i in range(len(fig1.data)):
                            step = {"method": "update", "args": [{"visible": [False] * len(fig1.data)}]}
                            step["args"][0]["visible"][i] = True
                            steps.append(step)
                        sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                        fig1.update_layout(showlegend=False, sliders=sliders)
                        fig1.show()
                    elif user_inp == "trace2":
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scattergl(showlegend=False, visible=True, line=dict(color="mediumseagreen", width=1.5),
                                                    x=x_array, y=sum_array))
                        fig2.update_xaxes(range=[x_array.min()-10, x_array.max()+10])
                        fig2.update_yaxes(range=[y_min_val, sum_array.max() + 0.05*sum_array.max()])
                        for k in range(len(new_xcoords2)):
                            fig2.add_shape(type="line", xref='x', yref='y',
                                           x0=new_xcoords2[k], y0=810, x1=new_xcoords2[k], y1=new_ycoords2[k],
                                           line=dict(color="cornflowerblue", width=1))
                        fig2.update_layout(title_text=f"Atomic lines of {name2} in range: {new_xcoords2.min():.2f}-{new_xcoords2.max():.2f} nm<br>and summed counts of the experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                        x_range2 = np.linspace(new_xcoords2.min()-20, new_xcoords2.max()+20, 60000)
                        for step in np.arange(0, 1, 0.01):
                            fig2.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                        name=f"σ = {step}", x=x_range2, y=SumGauss(x_range2, new_xcoords2, step, new_ycoords2)))
                        fig2.data[4].visible = True
                        steps = []
                        total = len(fig2.data)
                        for i in range(len(fig2.data)):
                            step = {"method": "update", "args": [{"visible": [False] * len(fig2.data)}]}
                            step["args"][0]["visible"][i] = True
                            steps.append(step)
                        sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                        fig2.update_layout(showlegend=False, sliders=sliders)
                        fig2.show()
                    elif user_inp == "trace3":
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scattergl(showlegend=False, visible=True, line=dict(color="mediumseagreen", width=1.5),
                                                    x=x_array, y=sum_array))
                        fig3.update_xaxes(range=[x_array.min()-10, x_array.max()+10])
                        fig3.update_yaxes(range=[y_min_val, sum_array.max() + 0.05*sum_array.max()])
                        for k in range(len(new_xcoords3)):
                            fig3.add_shape(type="line", xref='x', yref='y',
                                           x0=new_xcoords3[k], y0=810, x1=new_xcoords3[k], y1=new_ycoords3[k],
                                           line=dict(color="cornflowerblue", width=1))
                        fig3.update_layout(title_text=f"Atomic lines of {name3} in range: {new_xcoords3.min():.2f}-{new_xcoords3.max():.2f} nm<br>and summed counts of the experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                        x_range3 = np.linspace(new_xcoords3.min()-20, new_xcoords3.max()+20, 60000)
                        for step in np.arange(0, 1, 0.01):
                            fig3.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                        name=f"σ = {step}", x=x_range3, y=SumGauss(x_range3, new_xcoords3, step, new_ycoords3)))
                        fig3.data[4].visible = True
                        steps = []
                        total = len(fig3.data)
                        for i in range(len(fig3.data)):
                            step = {"method": "update", "args": [{"visible": [False] * len(fig3.data)}]}
                            step["args"][0]["visible"][i] = True
                            steps.append(step)
                        sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                        fig3.update_layout(showlegend=False, sliders=sliders)
                        fig3.show()
                    elif user_inp == "trace4":
                        fig4 = go.Figure()
                        fig4.add_trace(go.Scattergl(showlegend=False, visible=True, line=dict(color="mediumseagreen", width=1.5),
                                                    x=x_array, y=sum_array))
                        fig4.update_xaxes(range=[x_array.min()-10, x_array.max()+10])
                        fig4.update_yaxes(range=[y_min_val, sum_array.max() + 0.05*sum_array.max()])
                        for k in range(len(new_xcoords4)):
                            fig4.add_shape(type="line", xref='x', yref='y',
                                           x0=new_xcoords4[k], y0=810, x1=new_xcoords4[k], y1=new_ycoords4[k],
                                           line=dict(color="cornflowerblue", width=1))
                        fig4.update_layout(title_text=f"Atomic lines of {name4} in range: {new_xcoords4.min():.2f}-{new_xcoords4.max():.2f} nm<br>and summed counts of the experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                        x_range4 = np.linspace(new_xcoords4.min()-20, new_xcoords4.max()+20, 60000)
                        for step in np.arange(0, 1, 0.01):
                            fig4.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                        name=f"σ = {step}", x=x_range4, y=SumGauss(x_range4, new_xcoords4, step, new_ycoords4)))
                        fig4.data[4].visible = True
                        steps = []
                        total = len(fig4.data)
                        for i in range(len(fig4.data)):
                            step = {"method": "update", "args": [{"visible": [False] * len(fig4.data)}]}
                            step["args"][0]["visible"][i] = True
                            steps.append(step)
                        sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                        fig4.update_layout(showlegend=False, sliders=sliders)
                        fig4.show()
                    elif user_inp == "trace5":
                        fig5 = go.Figure()
                        fig5.add_trace(go.Scattergl(showlegend=False, visible=True, line=dict(color="mediumseagreen", width=1.5),
                                                    x=x_array, y=sum_array))
                        fig5.update_xaxes(range=[x_array.min()-10, x_array.max()+10])
                        fig5.update_yaxes(range=[y_min_val, sum_array.max() + 0.05*sum_array.max()])
                        for k in range(len(new_xcoords5)):
                            fig5.add_shape(type="line", xref='x', yref='y',
                                           x0=new_xcoords5[k], y0=810, x1=new_xcoords5[k], y1=new_ycoords5[k],
                                           line=dict(color="cornflowerblue", width=1))
                        fig5.update_layout(title_text=f"Atomic lines of {name5} in range: {new_xcoords5.min():.2f}-{new_xcoords5.max():.2f} nm<br>and summed counts of the experimental spectrum",
                                           title_font_size=30, xaxis_title="Wavelength [nm]", yaxis_title="Relative intensity")
                        x_range5 = np.linspace(new_xcoords5.min()-20, new_xcoords5.max()+20, 60000)
                        for step in np.arange(0, 1, 0.01):
                            fig5.add_trace(go.Scattergl(visible=False, line=dict(color="red", width=1.3),
                                                        name=f"σ = {step}", x=x_range5, y=SumGauss(x_range5, new_xcoords5, step, new_ycoords5)))
                        fig5.data[4].visible = True
                        steps = []
                        total = len(fig5.data)
                        for i in range(len(fig5.data)):
                            step = {"method": "update", "args": [{"visible": [False] * len(fig5.data)}]}
                            step["args"][0]["visible"][i] = True
                            steps.append(step)
                        sliders = [dict(active=4, currentvalue={"prefix": "σ: "}, pad={"t": 50}, steps=steps)]
                        fig5.update_layout(showlegend=False, sliders=sliders)
                        fig5.show()
        except Exception as e:
            print(f"Errore nell'analisi combinata: {e}")
