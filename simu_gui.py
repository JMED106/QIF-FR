"""
    SimuGUI - Graphical Tool to manage simulations.
    Copyright (C) 2017  Jose M. Esnaola-Acebes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import Queue
import ctypes as c
import logging
import multiprocessing
import time
import os
from sconf import now

try:
    import gi
except ImportError:
    logging.exception("Requires pygobject to be installed.")
    gi = None
    exit(1)

try:
    gi.require_version("Gtk", "3.0")
except ValueError:
    logging.exception("Requires gtk3 development files to be installed.")
except AttributeError:
    logging.exception("pygobject version too old.")

try:
    gi.require_version("Gdk", "3.0")
except ValueError:
    logging.exception("Requires gdk development files to be installed.")
except AttributeError:
    logging.exception("pygobject version too old.")

try:
    gi.require_version("GObject", "2.0")
except ValueError:
    logging.exception("Requires GObject development files to be installed.")
except AttributeError:
    logging.exception("pygobject version too old.")
try:
    from gi.repository import Gtk, GObject
except (ImportError, RuntimeError):
    logging.exception("Requires pygobject to be installed.")

import numpy as np
import matplotlib

matplotlib.use("Gtk3Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar

logging.getLogger('gui').addHandler(logging.NullHandler())
TARGET_TYPE_URI_LIST = 0


# Global methods
def flatten(foo):
    for x in foo:
        if hasattr(x, '__iter__'):
            for y in flatten(x):
                yield y
        else:
            yield x


def add_filters(dialog, flter=('text',), extensions=(None,)):
    for ext in extensions:
        if ext:
            fil = Gtk.FileFilter()
            fil.set_name(ext)
            fil.add_pattern("*" + ext)
            dialog.add_filter(fil)

    if 'text' in flter:
        filter_text = Gtk.FileFilter()
        filter_text.set_name("Text files")
        filter_text.add_mime_type("text/plain")
        dialog.add_filter(filter_text)

    if 'python' in flter:
        filter_py = Gtk.FileFilter()
        filter_py.set_name("Python files")
        filter_py.add_mime_type("text/x-python")
        dialog.add_filter(filter_py)

    filter_any = Gtk.FileFilter()
    filter_any.set_name("Any files")
    filter_any.add_pattern("*")
    dialog.add_filter(filter_any)


class MainGui:
    """ Main window which will host a simulation (can be extended to host more), and will
        call to other Widgets such as plots.
    """

    def __init__(self, data, simulation, tab_sets=(None,)):
        """ A set of data, appropriately formatted is needed, and a callable object
        """

        self.data = data
        self.sfunc = simulation

        self.logger = logging.getLogger('gui.MainGui')
        scriptpath = os.path.realpath(__file__)
        scriptdir = os.path.dirname(scriptpath)

        # GUI interface is loaded from a Glade file
        self.builder = Gtk.Builder()
        self.builder.add_from_file("%s/simu_win_1.1.glade" % scriptdir)

        # We identify windows
        self.window = self.builder.get_object("window1")
        self.window.connect("delete-event", self._on_exit_clicked)
        self.exit = True

        signals = {"gtk_main_quit": self._on_exit_clicked,
                   "on_Update_clicked": self.dummy,
                   "on_Pause_clicked": self._on_pause_clicked,
                   "on_Stop_clicked": self._on_stop_clicked,
                   "on_Quit_clicked": self._on_exit_clicked,
                   "on_entry_activate": self._on_value_changed,
                   "on_combo_changed": self._on_combo_changed,
                   "on_add_clicked": self._on_add_clicked,
                   "on_value_changed": self._on_value_changed,
                   "on_menu_new_activate": self.newsimulation,
                   "on_menu_save_activate": self.save_results,
                   "on_menu_open_activate": self._load_ic,
                   "on_menu_quit_activate": self._on_exit_clicked,
                   "on_menu_newplot_activate": self.newplot,
                   "on_menu_colorplot_activate": self.newcolorplot,
                   "on_menu_raster_activate": self.newraster,
                   "on_menu_vsnapshot_activate": self._take_snapshot,
                   "on_menu_freqs_activate": self._measure_freqs,
                   "on_menu_about_activate": self._about,
                   }

        self.builder.connect_signals(signals)
        self._space = ""

        self.about_dialog = self.builder.get_object("about_dialog")
        self.pause_button = self.builder.get_object("Pause")
        self.pause_button.set_label('Start')
        # If the simulated system accepts colorplots we enable the colorplot menu button
        if data.nf:
            menu_colorplot = self.builder.get_object('menu_colorplot')
            menu_colorplot.set_sensitive(True)
        if set(self.data.systems).intersection({'qif-fr', 'if-fr', 'eif-fr', 'qif-nf', 'if-nf', 'eif-nf'}):
            menu_raster = self.builder.get_object('menu_raster')
            menu_raster.set_sensitive(True)
            menu_snapshot = self.builder.get_object('menu_vsnapshot')
            menu_snapshot.set_sensitive(True)
            menu_freqs = self.builder.get_object('menu_freqs')
            menu_freqs.set_sensitive(True)

        # Set up of the parameters tab (this is done always)
        # Create the list of the combobox with the parameters in data
        self.notebook = self.builder.get_object('categories')
        combo = self.find_widget_down(self.window, "GtkComboBoxText")  # Find the combobox
        self.listbox = self.find_widget_down(self.window, "GtkListBox")  # FInd the listbox
        self.elements = self.extract_tags(self.data.opts['parameters'])  # Create the list
        store = self.update_tag_list(self.elements)  # Create the store
        self.update_combobox(combo, store)  # Update the combobox

        # Prepare multiprocessing framework, we need an input queue and an output queue
        self.q_in = multiprocessing.Queue()
        self.q_out = multiprocessing.Queue()
        self.multi_var = {}  # An additional object of shared memory, for plotting, saving, etc.
        self.logger.debug("Setting up shared memory...")
        for var in self.data.vars.keys():
            # self.logger.debug("\tVariable '%s' is of type: %s" % (var, type(self.data.vars[var])))
            if isinstance(self.data.vars[var], type(np.array([0]))):
                shape = np.shape(self.data.vars[var])
                dim = np.size(shape)
                # self.logger.debug("\t - Identified as array (f). Dimension: %s" % dim)
                if dim > 1:
                    length = shape[0] * shape[1]
                    array = self.data.vars[var] * 1.0
                    # self.logger.debug("\t\tTotal length is %d" % length)
                    self.multi_var[var] = np.frombuffer(multiprocessing.Array(c.c_double, length).get_obj()).reshape(
                        shape)
                    self.data.vars[var] = self.multi_var[var]
                    self.data.vars[var] = array * 1.0
                    del array
                else:
                    self.multi_var[var] = multiprocessing.Array('f', self.data.vars[var])
            elif isinstance(self.data.vars[var], int):
                # self.logger.debug(" - Identified as value (i).")
                self.multi_var[var] = multiprocessing.Value('i', self.data.vars[var])
            elif isinstance(self.data.vars[var], float):
                # self.logger.debug(" - Identified as value (i).")
                self.multi_var[var] = multiprocessing.Value('f', self.data.vars[var])

        self.Save = data.Save(data.opts, self.multi_var)
        self.simu_thread = None
        self.graphs = []

        # Populate the combo boxes with all the available parameters (optional):
        if self.data.all:
            combo = self.builder.get_object("comboboxtext1")
            elements = len(self.elements)
            for k in xrange(elements):
                # self.logger.debug("Element %d/%d." % (k, elements))
                combo.set_active(0)
                self._on_combo_changed(combo)
                if k != elements - 1:
                    combo = self._on_add_clicked(None)

        # Additional tabs set up (optional)
        # Function that creates the set up using parameters in tab_set
        try:
            tab_sets = list(tab_sets)
        except TypeError:
            tab_sets = [tab_sets]
        self.tab = {}
        for tab in tab_sets:
            if tab:
                self.tab[tab] = {'default': data.opts[tab].copy(), 'custom': data.opts[tab].copy()}
                self._new_tab(data.opts[tab], tab, self.notebook)
                self.tab[tab]['auto'] = False

    def __call__(self):
        # GObject.threads_init()
        self.window.show_all()
        Gtk.main()
        return self.exit

    def _new_tab(self, opts, label, notebook):
        # Function that takes a list of parameters and builds a tab with the options
        self.logger.debug("Setting up additional tab: %s..." % label)
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.set_border_width(15)
        page.set_name(label)
        grid = Gtk.Grid(row_spacing=10)
        page.pack_start(grid, True, True, 0)
        groups = {}
        # Arrange elements by type
        for k, opt in enumerate(opts.keys()):
            typ = type(opts[opt])
            if typ not in groups:
                groups[typ] = {opt: opts[opt]}
            else:
                groups[typ].update({opt: opts[opt]})
                # self.logger.debug("Type of element %d, '%s' is %s." % (k, opt, typ))
                # grid.attach(Gtk.Label(opt.capitalize() + ':', xalign=0, yalign=0.5), 0, k, 1, 1)
        # self.logger.debug("Elements: %s" % groups)
        # Set customize widgets for each type
        count = 0
        k = 0
        adjustment = None
        digits = None

        for typ in groups.keys():
            # self.logger.debug("Elements of type %s: %s" % (typ, groups[typ]))
            for k, opt in enumerate(groups[typ].keys()):
                lbl = Gtk.Label(opt.capitalize() + ':', xalign=0, yalign=0.5)
                lbl.set_hexpand(True)
                grid.attach(lbl, 0, count + k, 1, 1)

                if typ in (int, float):
                    if typ is int:  # SpinButton with integer values
                        adjustment = Gtk.Adjustment(0, -100, 1E8, 1, 5, 0)
                        digits = 0
                    elif typ is float:  # SpinButton with float values
                        adjustment = Gtk.Adjustment(0.0, -10.0, 100.0, 0.01, 1, 0)
                        digits = 2
                    widget = Gtk.SpinButton(adjustment=adjustment, value=0, numeric=True, digits=digits)
                    widget.set_value(opts[opt])
                    widget.connect("value-changed", self._on_value_changed)
                    widget.set_alignment(1.0)
                    widget.connect("activate", self._on_value_changed)
                elif typ in (str, list):
                    widget = Gtk.Entry()
                    if typ is list:
                        text = ", ".join(map(str, opts[opt]))
                        widget.set_text(text)
                    else:
                        widget.set_text(opts[opt])
                    widget.connect('activate', self._on_entry_activate)
                elif typ is bool:
                    widget = Gtk.Switch(state=opts[opt])
                    widget.set_hexpand(True)
                    widget.set_halign(Gtk.Align.END)
                    widget.connect("notify::active", self._on_switch_activated)
                else:
                    self.logger.warning("%s for element %s not implemented." % (typ, opts[opt]))
                    exit(-1)

                widget.set_hexpand(True)
                grid.attach(widget, 1, count + k, 1, 1)
                widget.set_name(opt)
            count += (k + 1)

        # Two buttons: one for applying changes the other one to reset the class
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        button_box.set_margin_top(10)
        page.pack_end(button_box, True, False, 0)
        reset_button = Gtk.Button.new_from_icon_name(Gtk.STOCK_CLEAR, 6)
        reset_button.connect("clicked", self._reset)
        button_box.pack_end(reset_button, False, False, 0)
        apply_button = Gtk.Button.new_from_icon_name(Gtk.STOCK_APPLY, 6)
        apply_button.connect("clicked", self._apply)
        button_box.pack_end(apply_button, False, False, 0)

        play_button = Gtk.Switch()
        play_button.set_active(False)
        play_button.connect("notify::active", self._auto)
        button_box.pack_end(play_button, False, False, 0)

        notebook.append_page(page, Gtk.Label(label.capitalize()))

    def _explore_tree(self, widget, attribute='name'):
        """ Function to completely explore the widgets of the GUI"""
        self._space += "-"
        for child in widget.get_children():
            if attribute == 'name':
                self.logger.debug("%s>  %s" % (self._space, child.get_name()))
            elif attribute == 'type':
                self.logger.debug("%s>  %s" % (self._space, child))
            try:
                self._explore_tree(child)
            except AttributeError:
                self._space = self._space[:-1]
        self._space = self._space[:-1]

    def _about(self, button):
        self.about_dialog.set_transient_for(self.window)
        self.about_dialog.run()
        self.about_dialog.hide()

    @staticmethod
    def find_widget_down(source, target):
        """ Method to find a successor child of a given source widget"""
        for child in source.get_children():
            if child.get_name() == target:
                # logging.debug("Target child found.")
                return child
            else:
                try:
                    targetchild = MainGui.find_widget_down(child, target)
                    if targetchild:
                        return targetchild
                except AttributeError:
                    # logging.debug("Target child not found in this branch.")
                    pass

    @staticmethod
    def find_widget_up(source, target):
        """ Method for finding an ancestor widget from a source widget."""
        parent = source
        while parent.get_name() != target:
            parent = parent.get_parent()
            try:
                name = parent.get_name()
                # logging.debug("Parent name: %s." % name)
            except AttributeError:
                # logging.debug("Target widget %s not in this branch." % target)
                return None
        return parent

    @staticmethod
    def identify_tab(notebook, name=True):
        page = notebook.get_current_page()
        tab = notebook.get_nth_page(page)
        tab_name = tab.get_name()
        if name:
            return tab_name
        else:
            return tab

    @staticmethod
    def extract_tags(dictionary, types=(int, float,), onlytags=True):
        """ Method to extract keys from a dictionary provided their value is either a float number or
            an integer number.
        """
        tags = []
        keys = dictionary.keys()
        for key in keys:
            if isinstance(dictionary[key], types) and not isinstance(dictionary[key], bool):
                if onlytags:
                    tags.append(key)
                else:
                    tags.append([key, np.shape(dictionary[key])])
        return tags

    def dummy(self, event):
        pass

    def _take_snapshot(self, event):
        self.logger.debug('Button %s pressed' % event)
        self.q_in.put({'controls': {'vsnapshot': True}})

    def _measure_freqs(self, event):
        self.logger.debug('Button %s pressed' % event)
        self.q_in.put({'controls': {'neuronf': True}})

    def _on_exit_clicked(self, event, *args, **kwargs):
        """ Event function to quit the programm, it should take care of every opened process."""
        self.logger.debug('Button %s pressed' % event)
        self._on_stop_clicked(None)
        time.sleep(0.1)
        self.q_in.close()
        self.q_out.close()
        if self.simu_thread:
            if self.simu_thread.is_alive():
                self.simu_thread.terminate()
                self.logger.debug('Thread terminated.')
        for graph in self.graphs:
            if graph:
                graph.PLOT = False

        self.exit = True
        Gtk.main_quit()

    def _on_pause_clicked(self, event):
        self.logger.debug('Button %s pressed' % event)
        self.data.opts['controls']['pause'] = not self.data.opts['controls']['pause']
        self.q_in.put({'controls': {'pause': self.data.opts['controls']['pause']}})
        if self.data.opts['controls']['pause']:
            self.pause_button.set_label('Resume')
        else:
            self.pause_button.set_label('Pause')

    def _on_stop_clicked(self, event):
        self.logger.debug('Button %s pressed' % event)
        self.q_in.put({'controls': {'stop': True}})
        self.pause_button.set_label('Start')

    def _on_combo_changed(self, combo):
        """ Changing the combobox will change the value of the entry at the right side of the combo box.
            Therefore we need to know in which row or box is the selected combo box. """
        # self.logger.debug('Element on %s modified' % combo.get_name())
        # Let's get the name of the parameter and its value
        element = combo.get_active_text()
        value = self.data.opts['parameters'][element]
        # Let's get the listboxrow where the combo is located
        listboxrow = self.find_widget_up(combo, 'GtkListBoxRow')
        # self.logger.debug("The %s is in the %s" % (combo.get_name(), listboxrow.get_name()))

        # The entry/spinbox next to the combobox
        entry = self.find_widget_down(listboxrow, 'GtkSpinButton')
        if entry:
            # Set the value in the entry
            entry.set_value(value)
        else:
            return 1

    def _on_entry_activate(self, entry):
        name = entry.get_name()
        self.logger.debug('Element on %s modified' % name)
        tab = self.identify_tab(self.notebook)
        self.logger.debug("%s is in %s tab" % (name, tab))
        if name != "GtkEntry":  # These entries are in optional tabs
            # Find the tab name
            value = entry.get_text()
            if ',' in value:
                value = value.split(',')
                try:
                    value = [int(val) for val in value]
                except ValueError:
                    value = [float(val) for val in value]
            elif value.isdigit():
                value = int(value)
            elif isinstance(self.tab[tab]['custom'][name], float):
                try:
                    value = float(value)
                except ValueError:
                    self.logger.error("Bad format introduced in %s, at %s" % (name, tab))
            if isinstance(self.tab[tab]['custom'][name], type(value)):
                self.tab[tab]['custom'][name] = value
            elif isinstance(self.tab[tab]['custom'][name], list):
                if isinstance(self.tab[tab]['custom'][name][0], type(value)):
                    self.tab[tab]['custom'][name] = [value]
                else:
                    self.logger.error("Bad format introduced in %s, at %s" % (name, tab))
            else:
                self.logger.error("Bad format introduced in %s, at %s" % (name, tab))
            if self.tab[tab]['auto']:
                return self._apply(None, tab)
            else:
                return 0
        # We must know the value in the combobox next to the entry
        listboxrow = self.find_widget_up(entry, 'GtkListBoxRow')
        self.logger.debug("The %s is in the %s" % (entry.get_name(), listboxrow.get_name()))
        combo = self.find_widget_down(listboxrow, 'GtkComboBoxText')
        if combo:
            element = combo.get_active_text()
        else:
            return 1
        if element:
            tipo = type(self.data.opts[tab][element])
            self.logger.debug("Type of %s is %s" % (element, tipo))
        else:
            return 1
        # Fucking comma instead of dot
        value = entry.get_text()
        if tipo != str:
            self.data.opts[tab][element] = tipo(value.replace(',', '.'))
        else:
            self.data.opts[tab][element] = value

        self.q_in.put({tab: {element: value}})

    def _on_value_changed(self, spinbox):
        name = spinbox.get_name()
        # self.logger.debug('Element on %s modified' % name)
        tab = self.identify_tab(self.notebook)
        # self.logger.debug("%s is in %s tab" % (name, tab))
        if name != "GtkSpinButton":
            # Find the tab name
            value = spinbox.get_value()
            self.tab[tab]['custom'][name] = value
            if self.tab[tab]['auto']:
                return self._apply(None, tab)
            else:
                return 0
        else:
            # We must know the value in the combobox next to the entry
            listboxrow = self.find_widget_up(spinbox, 'GtkListBoxRow')
            # self.logger.debug("The %s is in the %s" % (spinbox.get_name(), listboxrow.get_name()))
            combo = self.find_widget_down(listboxrow, 'GtkComboBoxText')
            if combo:
                element = combo.get_active_text()
            else:
                return 1
            value = spinbox.get_value()
            if element:
                self.data.opts[tab][element] = value
                # self.logger.debug("Element %s changed to %s" % (element, str(value)))
                self.q_in.put({tab: {element: value}})

    def _on_switch_activated(self, switch):
        name = switch.get_name()
        # Find the tab name
        tab = self.identify_tab(self.notebook)
        self.logger.debug("%s is in %s tab" % (name, tab))
        self.tab[tab]['custom'][name] = switch.get_active()
        return 0

    def _reset(self, button):
        # TODO: reset values in all the grid
        name = button.get_name()
        tab = self.identify_tab(self.notebook)
        self.logger.debug("%s pressed. Settings in %s to be reset." % (name, tab))
        self.q_in.put({tab: self.tab[tab]['default']})
        return 0

    def _apply(self, button, tab=None, name='None'):
        if button:
            name = button.get_name()
            tab = self.identify_tab(self.notebook)

        self.logger.debug("%s pressed. Settings in %s to be applied." % (name, tab))
        self.data.opts[tab].update(self.tab[tab]['custom'])
        self.q_in.put({tab: self.tab[tab]['custom']})
        return 0

    def _auto(self, button, gparam):
        name = button.get_name()
        tab = self.identify_tab(self.notebook)
        if button.get_active():
            self.logger.debug("%s pressed. Settings in %s are now automatically changing." % (name, tab))
            self.tab[tab]['auto'] = True
        else:
            self.logger.debug("%s pressed. Settings in %s must be applied manually." % (name, tab))
            self.tab[tab]['auto'] = False
        self.data.opts[tab].update(self.tab[tab]['custom'])
        self.q_in.put({tab: self.tab[tab]['custom']})
        return 0

    def _on_add_clicked(self, button):
        """ Add a new row to be able to modify another parameter"""
        if len(self.elements) > 1:
            # if button:
            #     self.logger.debug('Element on %s modified' % button.get_name())
            # We freeze the previous combo box after checking an element is selected
            prev_row = self.listbox.get_children()[-1]
            prev_combo = self.find_widget_down(prev_row, "GtkComboBoxText")
            element = prev_combo.get_active_text()
            if not element:
                self.logger.warning("The previous combo box has not been used yet.")
                return 1
            else:
                # We also remove the previously used element from the new list (to avoid repetition)
                prev_element = prev_combo.get_active_text()
                self.elements.remove(prev_element)
                model = self.update_tag_list(self.elements)

            prev_combo.set_sensitive(False)
            newbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)  # The box containing the widgets
            self.listbox.insert(newbox, -1)  # The new listboxrow
            row = self.listbox.get_children()[-1]
            row.set_selectable(False)

            # First child
            combobox = Gtk.ComboBoxText.new()  # The combobox
            self.update_combobox(combobox, model)
            combobox.connect("changed", self._on_combo_changed)
            newbox.pack_start(combobox, True, True, padding=4)

            # Second child
            adjustment = Gtk.Adjustment(0, -100, 1E8, 0.01, 1, 0)
            value = Gtk.SpinButton(adjustment=adjustment, value=0.00, digits=4, numeric=True, width_chars=8,
                                   max_width_chars=8, max_length=8)
            value.set_alignment(1.0)
            value.connect("activate", self._on_value_changed)
            value.connect("value-changed", self._on_value_changed)
            newbox.pack_start(value, True, True, padding=4)

            self.window.show_all()
            return combobox

    def newplot(self, menu):
        """ Function that creates a new Gtk window with canvas where a Matplotlib plot is created.
            For that it uses the class Graph.
            It also asks for the variables to be plotted.
        """

        self.logger.debug('Element on %s modified' % menu.get_name())
        # Ask variables and range
        dialog = PlotDialog(self.data.vars, self.data.lims, parent=self.window)
        dialog.run()
        dialog.hide()
        if dialog.accept:
            title = dialog.plt_vars['y'] + ' vs. ' + dialog.plt_vars['x']
            if dialog.polar:
                self.logger.debug('Polar graph.')
                polar = 'polar'
            else:
                polar = None
            graph = Graph(self.multi_var, title=title, pvars=(dialog.plt_vars['x'], dialog.plt_vars['y']),
                          store=dialog.store, lims=self.data.lims, polar=polar, pop=dialog.n, fixed=dialog.fixed)
            graph.nsteps = self.data.nsteps
            graph.tau = self.data.faketau
            graph.ax.set_xlim(dialog.lim['x'])
            graph.ax.set_ylim(dialog.lim['y'])
            graph.show_all()
            self.graphs.append(graph)
        dialog.destroy()

    def newcolorplot(self, menu):
        """ Function that creates a new Gtk window with canvas where a Matplotlib plot is created.
            For that it uses the class Graph.
            It also asks for the variables to be plotted.
        """

        self.logger.debug('Element on %s modified' % menu.get_name())
        # Ask variables and range
        dialog = PlotDialog(self.data.vars, self.data.lims, parent=self.window)
        dialog.polar_check.set_visible(False)
        dialog.run()
        dialog.hide()
        if dialog.accept:
            title = "Color plot of " + dialog.plt_vars['y'] + ' along ' + dialog.plt_vars['x']
            graph = ColorMap(self.multi_var, title=title, labels=(dialog.plt_vars['x'], dialog.plt_vars['y']),
                             store=dialog.store, lims=self.data.lims)
            if set(self.data.systems).intersection('qif-nf', 'if-nf', 'eif-nf'):
                graph.initplot(self.data.fr.length, **self.data.opts)
            else:
                graph.initplot(self.data.nsteps, **self.data.opts)
            graph.zlim = dialog.lim['y']
            # graph.ax.set_xlim(dialog.lim['x'])
            # graph.ax.set_ylim(dialog.lim['y'])
            graph.show_all()
            self.graphs.append(graph)
        dialog.destroy()

    def newraster(self, menu):
        """ Function that creates a new Gtk window with canvas where a Matplotlib plot is created.
            For that it uses the class Graph.
            It also asks for the variables to be plotted.
        """

        self.logger.debug('Element on %s modified' % menu.get_name())
        # Ask variables and range only for spiking neuron ralated variables
        sp_vars = {'tfr', 'sp_re', 'sp_ri'}
        variables = {}
        for sp_var in sp_vars:
            data = self.data.vars.get(sp_var, None)
            if data is not None:
                variables[sp_var] = data
        dialog = PlotDialog(variables, self.data.lims, parent=self.window, raster=True)
        if self.data.nf:
            dialog.polar_check.set_label("All.")
        else:
            dialog.polar_check.set_visible(False)
        dialog.run()
        dialog.hide()
        if dialog.accept:
            self.data.opts['raster']['start'] = True
            if not dialog.polar:
                self.data.opts['raster']['pop'] = dialog.n
            self.q_in.put({'raster': {'start': self.data.opts['raster']['start'],
                                      'pop': self.data.opts['raster']['pop']}})
            title = dialog.plt_vars['y'] + ' vs. ' + dialog.plt_vars['x']
            raster = RasterPlotC(self.data, self.multi_var, self.q_in, self.q_out, title=title,
                                labels=(dialog.plt_vars['x'], dialog.plt_vars['y']),
                                lims=self.data.lims)
            raster.initplot(**self.data.opts)
            raster.show_all()
            self.graphs.append(raster)

        dialog.destroy()

    def newsimulation(self, menu):
        """ Function to send a job to the child process. TO BE FIXED Stout problem."""
        self.logger.debug('Element on %s modified' % menu.get_name())
        try:
            while not self.q_out.empty():
                data = self.q_out.get_nowait()
                if data.get('opts', False):
                    self.data.opts = data['opts']
                elif data.has_key('m_e'):
                    self.data.m_e = data['m_e']
                    self.data.m_i = data['m_i']
                    self.data.spk_e = data['spk_e']
                    self.data.spk_i = data['spk_i']
                    try:
                        self.data.spk_e_mod = data['spk_e_mod']
                        self.data.spk_i_mod = data['spk_i_mod']
                    except:
                        pass
        except Queue.Empty:
            self.logger.debug("No data in the output queue.")
        self.data.opts['controls'].update({'stop': False, 'pause': True, 'exit': False})
        if self.simu_thread:
            if self.simu_thread.is_alive():
                self.simu_thread.terminate()
                del self.simu_thread

        self.simu_thread = multiprocessing.Process(None, self.sfunc,
                                                   args=(self.data, self.multi_var, self.q_in, self.q_out))
        self.simu_thread.start()

    @staticmethod
    def update_combobox(combo, elements):
        combo.set_model(elements)
        combo.set_entry_text_column(0)

    @staticmethod
    def update_tag_list(elements, listelements=None):
        """ Method that updates a liststore extracting data from a list. If the list has more than one value per
            element, it tries to cerate a liststore of multiple columns. """
        length = 0
        if listelements is not None:
            del listelements
        # A list is required
        if type(elements) is list:
            # If the elements of the list are lists: identify the longest element and extract the types
            if type(elements[0]) is list:
                # Find the longest one
                length = len(list(flatten(elements[0])))
                idx = 0
                for k, ele in enumerate(elements):
                    if length < len(list(flatten(ele))):
                        length = len(list(flatten(ele)))
                        idx = k
                # Extract types
                typ = []
                for ele in list(flatten(elements[idx])):
                    typ.append(type(ele))
                element_store = Gtk.ListStore(*typ)
            else:
                element_store = Gtk.ListStore(type(elements[0]))
        else:
            logging.error("'elements' must be a list.")
            return -1
        # Populate the ListStore
        if elements:
            for element in elements:
                if type(element) is list:
                    element = list(flatten(element))
                    while len(element) < length:
                        element.append(1)
                    element_store.append(element)
                else:
                    element_store.append([element])
        return element_store

    def save_results(self, menu):
        if menu:
            self.logger.debug('Element on %s modified' % menu.get_name())

        # Pause simulation if it is running
        paused = False
        if not self.data.opts['controls']['pause']:
            paused = True
            self._on_pause_clicked(None)
        dialog = SaveDialog(self.window, self.data.vars, **self.data.opts)
        response = dialog.run()
        dialog.hide()
        if response == Gtk.ResponseType.OK:
            # If the simulation is not running take the available data from within
            data = self.data.opts
            if self.simu_thread:
                if self.simu_thread.is_alive():
                    try:
                        while not self.q_out.empty():
                            data = self.q_out.get_nowait()
                        print data.keys()
                    except Queue.Empty:
                        pass
            # Save initial conditions
            if dialog.ic:
                self.Save(path=dialog.save_path, choice=dialog.choice, save_ic=True, data=data)
            # Save the data using the SaveData class in self.data
            elif dialog.all:
                self.Save(variables=self.multi_var, path=dialog.save_path, choice='all')
            else:
                self.Save(variables=self.multi_var, path=dialog.save_path, choice=dialog.choice)
        dialog.destroy()
        # Resume simulation if it was running
        if paused:
            self._on_pause_clicked(None)

    def _load_ic(self, menu):
        # Open file dialog to choose initial conditions file
        dialog = Gtk.FileChooserDialog("Please choose a file", self.window, Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        add_filters(dialog, flter=('text', 'python'), extensions=('.npy',))
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_current_folder(self.data.opts['dir'])
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            load_path = dialog.get_filename()
            # Modify data object with the new conditions
            self.data = self.Save(path=load_path, load_ic=True)
            dialog.destroy()
            self._on_exit_clicked(None)
            self.exit = False
        elif response == Gtk.ResponseType.CANCEL:
            dialog.hide()
            dialog.destroy()


class PlotDialog(Gtk.Dialog):
    __gtype_name__ = 'PlotDialog'

    def __new__(cls, pvars, lims, parent=None, xactive=None, **kwargs):
        """This method creates and binds the builder window to class.

        In order for this to work correctly, the class of the main
        window in the Glade UI file must be the same as the name of
        this class."""

        app_path = os.path.dirname(__file__)
        try:
            builder = Gtk.Builder()
            builder.add_from_file(os.path.join(app_path, "plot_dialog.glade"))
        except GObject.GError:
            print "Failed to load XML GUI file plot_dialog.glade"
            return -1
        new_object = builder.get_object('plt_dialog')
        new_object.finish_initializing(builder, pvars, lims, parent, xactive, **kwargs)
        return new_object

    # noinspection PyAttributeOutsideInit
    def finish_initializing(self, builder, pvars, lims, parent=None, xactive=None, **kwargs):
        """Treat this as the __init__() method.

        Arguments pass in must be passed from __new__()."""

        # Add any other initialization here
        self.logger = logging.getLogger('gui.PlotDialog')
        self._builder = builder

        signals = {"on_plt_combo_changed": self._on_plt_combo_changed,
                   "on_plt_value_changed": self._on_plt_value_changed,
                   "on_pop_value_changed": self._on_pop_value_changed,
                   "on_cancel": self._on_cancel,
                   "on_accept": self._on_accept,
                   "on_polar": self._on_polar,
                   "on_fixed": self._on_fixed
                   }

        builder.connect_signals(signals)
        self.connect("delete-event", self._on_cancel)

        # Create the list of the combobox of the plotting dialog with the variables in data
        combox = self._builder.get_object("plt_combox")
        comboy = self._builder.get_object("plt_comboy")

        self.plt_vars = {'x': 't'}
        self.lim = {'x': lims['t'] * 1, 'y': [0, 1.0]}
        self._default_limits = lims
        self.polar = False
        self.fixed = False
        self.fixed_button = builder.get_object('check_fixed')

        self.n = None
        self.pop_spin = builder.get_object("pop")
        self.pop_lbl = builder.get_object("pop_label")
        xtargets = {'tfr', 't'}
        time_var = False
        target = 't'

        # If the plot is a neural field, then we can disable polar check button
        # We may change this button for using it as other purposes
        self.polar_check = builder.get_object("polar_check")
        self.raster = kwargs.get('raster', False)
        # If the parent is not the main window (to add another plot)
        if isinstance(pvars, Gtk.ListStore):
            self.store = pvars
            target = xactive
            self.set_modal(False)
            self.logger.debug("A liststore passed")
            # combox.set_sensitive(False)
        else:
            while time_var is False:
                target = xtargets.pop()
                time_var = pvars.get(target, False)
            var_type = type(time_var)
            self._var_elements = MainGui.extract_tags(pvars, types=(var_type,), onlytags=False)
            self.store = MainGui.update_tag_list(self._var_elements)

        MainGui.update_combobox(combox, self.store)
        MainGui.update_combobox(comboy, self.store)

        # Set default values
        combox.set_active(0)
        i = 0
        while combox.get_active_text() != target:
            i += 1
            combox.set_active(i)
        if i == 0:
            comboy.set_active(1)
        else:
            comboy.set_active(0)

        self._on_plt_combo_changed(combox)
        self._on_plt_combo_changed(comboy)

        # Link to the parent window
        if parent:
            self.logger.debug("Linking the dialog to the parent.")
            self.set_transient_for(parent)

        self.accept = False
        self.logger.debug("Dialog initialized.")

    def _on_plt_combo_changed(self, combo):
        """ Changing the combobox will set the variable tag to pass to the graphing class """
        name = combo.get_name()
        self.logger.debug('Element on %s modified' % name)
        # Let's get the name of the variable
        element = combo.get_active_text()
        # Set the value in the variables to be passed to the graph
        self.plt_vars[name[-1]] = element
        # Set the default range values of that element
        grid = combo.get_parent()
        row = grid.child_get_property(combo, 'top_attach')
        column = grid.child_get_property(combo, 'left_attach')
        min_entry = grid.get_child_at(column + 2, row)
        max_entry = grid.get_child_at(column + 3, row)
        try:
            min_entry.set_value(self._default_limits[element][0])
            max_entry.set_value(self._default_limits[element][1])
        except KeyError:
            min_entry.set_value(0)
            max_entry.set_value(1.0)
        self._on_plt_value_changed(min_entry)
        self._on_plt_value_changed(max_entry)

        # Show fixed option if <phi> is selected
        if name == 'plt_combox':
            if element == 'phi':
                self.fixed_button.show()
            else:
                self.fixed_button.hide()
        # Show pop spin and label in case the variable is a matrix
        # Identify the row in the list store
        n = 0
        try:
            for row in self.store:
                if row[:][0] == element:
                    n = row[:][2]
        except IndexError:
            n = 1
        if n > 2:
            self.logger.debug("Showing population selection widgets.")
            adj = Gtk.Adjustment(lower=0, upper=n, value=int(n / 2), step_increment=1)
            self.pop_spin.set_adjustment(adj)
            self.pop_spin.show()
            self.pop_lbl.show()
            self.n = int(n / 2)
        else:
            self.pop_spin.hide()
            self.pop_lbl.hide()
            self.n = None

    def _on_plt_value_changed(self, spinbox):
        name = spinbox.get_name()
        value = spinbox.get_value()
        self.logger.debug('Element on %s modified to %f' % (name, value))
        if name[1:] == 'min':
            self.lim[name[0]][0] = value
        else:
            self.lim[name[0]][1] = value

    def _on_pop_value_changed(self, spinbox):
        self.n = int(spinbox.get_value())

    def _on_accept(self, button):
        self.logger.debug('Button %s pressed.' % button.get_name())
        self.hide()
        self.accept = True

    def _on_cancel(self, button):
        self.logger.debug('Button %s pressed.' % button.get_name())
        self.hide()
        self.accept = False

    def _on_polar(self, tickbutton):
        self.logger.debug('TickButton %s pressed.' % tickbutton.get_name())
        self.polar = not self.polar
        if self.polar and self.raster:
            self.pop_spin.set_sensitive(False)
        else:
            self.pop_spin.set_sensitive(True)

    def _on_fixed(self, tickbutton):
        self.logger.debug('TickButton %s pressed.' % tickbutton.get_name())
        self.fixed = not self.fixed


class Graph(Gtk.Window):
    """ Gtk object containing a canvas plus some other widget, such as a toolbox."""

    def __init__(self, data, title='Matplotlib', size=(800, 500), pvars=('t', 're'), store=None,
                 lims=None, **kwargs):
        """ Initialization requires a memory shared data object (dictionary). And some key values
            representing the variables to be plotted.
        """
        Gtk.Window.__init__(self, title=title)
        self.logger = logging.getLogger('gui.Graph')
        self.window = self
        self.vars = [pvars]
        self.store = store
        self.lims = lims
        self.data = data
        self.tau = 1.0
        self.nsteps = 0

        self.set_default_size(*size)
        self.boxvertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.connect("delete-event", self._destroy)
        self.add(self.boxvertical)

        self.toolbar = Gtk.Toolbar()
        self.context = self.toolbar.get_style_context()
        self.context.add_class(Gtk.STYLE_CLASS_PRIMARY_TOOLBAR)
        self.boxvertical.pack_start(self.toolbar, False, False, 0)

        self.refreshbutton = Gtk.ToolButton(icon_name='media-playback-start')
        self.toolbar.insert(self.refreshbutton, 0)

        self.addbutton = Gtk.ToolButton(Gtk.STOCK_ADD)
        self.toolbar.insert(self.addbutton, 0)
        self.addbutton.connect("clicked", self._add_variable)

        self.box = Gtk.Box()
        self.boxvertical.pack_start(self.box, True, True, 0)

        # This can be put into a figure or a class ####################
        self.fig = plt.Figure(figsize=(10, 10), dpi=80)
        try:
            self.ax = self.fig.add_subplot(111, projection=kwargs['polar'])
        except KeyError:
            self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        ###############################################################
        self.box.pack_start(self.canvas, True, True, 0)

        self.toolbar2 = NavigationToolbar(self.canvas, self)
        self.boxvertical.pack_start(self.toolbar2, False, True, 0)

        self.statbar = Gtk.Statusbar()
        self.boxvertical.pack_start(self.statbar, False, True, 0)

        self.fig.canvas.mpl_connect('motion_notify_event', self._updatecursorposition)

        # Check the system: extended system?
        p_func = self._plot2dcont
        self.n = None
        self.fixed = False
        pop = kwargs.get('pop', None)
        if isinstance(pop, int):
            self.extended = True
            try:
                self.fixed = kwargs['fixed']
                if self.fixed:
                    p_func = self._plot3dfixed
                    self.logger.debug("Plotting profile of the neural field.")
                else:
                    p_func = self._plot3dcont
                    self.logger.debug("Plotting pop %d of the neural field." % pop)
            except KeyError:
                p_func = self._plot3dcont
                self.logger.debug("Plotting pop %d of the neural field." % pop)
            self.n = [pop]
        else:
            self.extended = False
            self.logger.debug("Plotting one of the population's variables.")

        if self.fixed:
            self.addbutton.set_sensitive(False)
            self.addbutton.set_visible(False)
        # Depending on the introduced data the plot is different
        self.refreshbutton.connect("clicked", self.run_dynamically, p_func)

        # Figure plotting variables
        self.xdata = np.linspace(0, 2.0, 10)
        self.ydata = self.xdata * 0.0
        self.plots = []
        self.p1, = self.ax.plot(self.xdata, self.ydata, animated=False)
        self.plots.append(self.p1)
        self.ax.grid(True)
        self.ax.set_xlabel(pvars[0], fontsize='20')
        self.ax.set_ylabel(pvars[1], fontsize='20')
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.PLOT = False

    def _resetplot(self):
        # self.ax.cla()
        # self.ax.set_xlim(0, 10)
        # self.ax.set_ylim(0, 10)
        self.ax.grid(True)

    # noinspection PyUnusedLocal
    def _destroy(self, *args):
        self.PLOT = False

    def run_dynamically(self, button, func):
        self.logger.debug('Button %s pressed.' % button.get_name())
        self.PLOT = not self.PLOT
        GObject.timeout_add(20, self.plot, func)

    def plot(self, func):
        while Gtk.events_pending():
            Gtk.main_iteration()
        func()
        return self.PLOT

    def _plot2dcont(self):
        """ It changes the data of both axis taking the information from the shared memory object."""
        for p, var in zip(self.plots, self.vars):
            if var[0] == 'tfr':
                tstep = self.data['frtstep'].value
            else:
                tstep = self.data['tstep'].value % self.nsteps
            if var[0] in ('t', 'tfr') and tstep != 0:
                ydata1 = self.data[var[1]][tstep:]
                ydata2 = self.data[var[1]][0:tstep]
                ydata = np.concatenate((ydata1, ydata2))
            else:
                ydata = self.data[var[1]][::]
            if var[0] == 'tfr':
                # TODO: change the sampling interval depending on the length of the array (fr of spiking neuron best
                # with 0)
                p.set_xdata(np.array(self.data[var[0]]) * self.tau)
                p.set_ydata(np.array(ydata) / self.tau)
                # self.logger.debug(np.array(ydata) / self.tau)
            else:
                p.set_xdata(np.array(self.data[var[0]][::50]) * self.tau)
                p.set_ydata(np.array(ydata[::50]) / self.tau)
        self.canvas.draw()

    def _plot3dcont(self):
        """ It changes the data of both axis taking the information from the shared memory object."""
        for k, (p, var) in enumerate(zip(self.plots, self.vars)):
            if var[0] == 'tfr':
                tstep = self.data['frtstep'].value
            else:
                tstep = self.data['tstep'].value % self.nsteps
            if var[0] in ('t', 'tfr') and tstep != 0:
                ydata1 = self.data[var[1]][tstep:, self.n[k]]
                ydata2 = self.data[var[1]][0:tstep, self.n[k]]
                ydata = np.concatenate((ydata1, ydata2))
            else:
                ydata = self.data[var[1]][::, self.n[k]]

            if var[0] == 'tfr':
                # TODO: change the sampling interval depending on the length of the array (fr of spiking neuron best with 0)
                p.set_xdata(np.array(self.data[var[0]]) * self.tau)
                p.set_ydata(np.array(ydata) / self.tau)
            else:
                p.set_xdata(np.array(self.data[var[0]][::50]) * self.tau)
                p.set_ydata(ydata[::50] / self.tau)
        self.canvas.draw()

    def _plot3dfixed(self):
        """ It changes the data of both axis taking the information from the shared memory object."""
        for k, (p, var) in enumerate(zip(self.plots, self.vars)):
            if var[1][0:2] == 'sp':
                tstep = self.data['frtstep'].value
            else:
                tstep = self.data['tstep'].value % self.nsteps
            ydata = self.data[var[1]][tstep]
            p.set_xdata(self.data[var[0]])
            p.set_ydata(ydata / self.tau)
        self.canvas.draw()

    def _updatecursorposition(self, event):
        """When cursor inside plot, get position and print to status-bar"""
        if event.inaxes:
            x = event.xdata
            y = event.ydata
            self.statbar.push(1, ("Coordinates:" + " x= " + str(round(x, 3)) + "  y= " + str(round(y, 3))))

    def _add_variable(self, button):
        self.logger.debug('Button %s pressed.' % button.get_name())
        self.logger.debug("Selecting another variable to plot.")
        dialog = DialogVar(self, self.store)
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            p, = self.ax.plot(self.xdata, self.ydata, animated=False)
            self.plots.append(p)
            self.vars.append([dialog.choice[0], dialog.choice[1]])
            if self.n:
                self.n.append(dialog.n)
        elif response == Gtk.ResponseType.CANCEL:
            pass

        dialog.destroy()
        # dialog = PlotDialog(self.store, self.lims, self.window, self.vars[0][0])
        self.logger.debug("Dialog created. Now running...")
        # dialog.run()
        # dialog.hide()
        # if dialog.accept:
        #     self.ax.set_xlim(dialog.lim['x'])
        #     self.ax.set_ylim(dialog.lim['y'])
        #     p, = self.ax.plot(self.xdata, self.ydata, animated=False)
        #     self.plots.append(p)


class ColorMap(Gtk.Window):
    """ Gtk object containing a canvas plus some other widget, such as a toolbox."""

    def __init__(self, data, title='Matplotlib', size=(800, 500), labels=('t', 're'), store=None, lims=None):
        """ Initialization requires a memory shared data object (dictionary). And some key values
            representing the variables to be plotted.
        """
        Gtk.Window.__init__(self, title=title)
        self.logger = logging.getLogger('gui.ColorMap')

        self.window = self
        self.labels = labels
        self.store = store
        self.lims = lims
        self.logger.debug("Limits are: %s" % self.lims[labels[1]])
        self.data = data
        self.logger.debug("Checking data size.\n\t\t\tX data: %s\n\t\t\tY data: %s" %
                          (data[labels[0]], np.shape(data[labels[1]])))
        # Variables that must be assigned first:
        self.tplot1 = 0
        self.tplot2 = 0

        self.set_default_size(*size)
        self.boxvertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.connect("delete-event", self._destroy)
        self.add(self.boxvertical)

        self.toolbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.boxvertical.pack_start(self.toolbox, False, False, 0)
        # In the upper part we locate a toolbox and Z range editor (all placed in a box)
        self.toolbar = Gtk.Toolbar()
        self.context = self.toolbar.get_style_context()
        self.context.add_class(Gtk.STYLE_CLASS_PRIMARY_TOOLBAR)
        self.toolbox.pack_start(self.toolbar, False, False, 0)

        self.refreshbutton = Gtk.ToolButton(icon_name='media-playback-start')
        self.toolbar.insert(self.refreshbutton, 0)
        self.refreshbutton.connect("clicked", self.run_dynamically)

        self.snapshot_button = Gtk.ToolButton(icon_name='camera-photo')
        self.toolbar.insert(self.snapshot_button, 0)
        self.snapshot_button.connect("clicked", self._take_snapshot)

        # Modifying ranges (two entries, one label, one button) We use a BOX for this
        self.range_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.toolbox.pack_end(self.range_box, False, False, 0)
        label = Gtk.Label('Z range: ')
        self.range_box.pack_start(label, False, False, 1)
        self.range_box.set_margin_top(5)
        self.range_box.set_margin_bottom(5)
        self.range_box.set_margin_right(5)
        self.vmin_entry = Gtk.Entry(placeholder_text='min', input_purpose='number',
                                    tooltip_text="Set the lower boundary of the color map's range.")
        self.range_box.pack_start(self.vmin_entry, False, False, 0)
        self.vmax_entry = Gtk.Entry(placeholder_text='max', input_purpose='number',
                                    tooltip_text="Set the upper boundary of the color map's range.")
        self.range_box.pack_start(self.vmax_entry, False, False, 0)
        self.set_range = Gtk.Button.new_from_icon_name(Gtk.STOCK_APPLY, 4)
        self.range_box.pack_start(self.set_range, False, False, 1)
        self.set_range.connect("clicked", self._set_ranges)

        # Modifying density of the plot
        self.density = 1
        spin = Gtk.Adjustment(lower=0.01, upper=1.0, step_increment=0.01, page_increment=0.1)
        self.density_spinentry = Gtk.SpinButton(max_length=8, width_chars=8, xalign=1, adjustment=spin,
                                                tooltip_text='Change density of the plotting.', input_purpose='number',
                                                climb_rate=0.01, digits=2, numeric=True)
        self.density_spinentry.set_value(1.0 / self.density)
        self.range_box.pack_start(self.density_spinentry, False, False, 0)
        self.density_spinentry.connect("activate", self._change_density)
        self.density_spinentry.connect("value-changed", self._change_density)

        self.box = Gtk.Box()
        self.boxvertical.pack_start(self.box, True, True, 0)

        # This can be put into a figure or a class ####################
        self.fig = plt.Figure(figsize=(10, 10), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        ###############################################################
        self.box.pack_start(self.canvas, True, True, 0)

        self.toolbar2 = NavigationToolbar(self.canvas, self)
        self.boxvertical.pack_start(self.toolbar2, False, True, 0)

        self.statbar = Gtk.Statusbar()
        self.boxvertical.pack_start(self.statbar, False, True, 0)

        self.fig.canvas.mpl_connect('motion_notify_event', self._updatecursorposition)

        # Figure plotting variables
        # TODO: implement general y axis

        # Limits of the colorplots
        self.xlabel = labels[0]
        if self.xlabel == 'tfr':
            self.tlabel = 'frtstep2'
        elif self.xlabel == 't':
            self.tlabel = 'tstep'
        else:
            self.logger.error("Colormap not implemented for the selected labels: (%s, %s)" % (labels[0], labels[1]))
            self.destroy()

        self.ylim = [-np.pi, np.pi]
        self.zlim = lims[labels[1]]
        self.zmax = None
        self.zmin = None
        self.cbar = None
        self.cycle = 0

        # Other variables
        self.PLOT = False
        self.reset = True
        self.counter = 0
        self.tries = 5

    # noinspection PyAttributeOutsideInit
    def initplot(self, nsteps, **kwargs):
        # Color plot initialization
        self.nsteps = nsteps
        self.n = kwargs['n']
        self.tau = kwargs['faketau']
        self.dt = self.data[self.xlabel][-1] - self.data[self.xlabel][-2]
        self.x = np.array(self.data[self.xlabel])
        self.y = np.linspace(-np.pi, np.pi, self.n + 1)
        z = np.zeros((self.n, 2))
        self.p = self.ax.pcolormesh(self.x[0:2], self.y, z, cmap=plt.get_cmap('gray'),
                                    vmin=self.zlim[0], vmax=self.zlim[1], animated=True)
        self.bar = plt.colorbar(self.p, ax=self.ax, fraction=0.1, pad=0.01)
        self.ax.set_xlabel(self.labels[0], fontsize='20')
        self.ax.set_ylabel(self.labels[1], fontsize='20')
        self.ax.set_yticks(np.arange(-np.pi, np.pi + np.pi / 2, np.pi / 2))
        y_ticklabels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
        self.ax.set_yticklabels(y_ticklabels, fontsize=14)
        self.ax.set_xlim(np.array([self.data['t'][0], self.data['t'][-1]]) * self.tau)
        self.fig.tight_layout()
        self.fig.canvas.draw()

        # We take the bounding boxes
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox.expanded(1.1, 1.1))

    def _resetplot(self):
        # Once a cycle is completed it erases the plot and starts again
        if self.tplot1 < self.cycle * self.nsteps:
            # self.logger.debug("Tplot1: %d\tTplot2: %d" % (self.tplot1, self.tplot2))
            self.tplot1 = self.cycle * self.nsteps
        # When time reaches the total_time we start the color plot again, and change limits
        # self.ax.clear()

        # Limits
        xlim = (self.cycle * self.nsteps * self.dt + np.array([self.data['t'][0], self.data['t'][-1]])) * self.tau
        self.ax.set_xlim(xlim)
        # Restore the region background
        self.fig.canvas.restore_region(self.bg)
        # self.fig.canvas.blit(self.ax.bbox)
        # self.fig.canvas.blit(self.bar.ax.bbox)
        # self.reset = True

    # noinspection PyUnusedLocal
    def _destroy(self, *args):
        self.PLOT = False

    def run_dynamically(self, button):
        self.logger.debug('Button %s pressed.' % button.get_name())
        self.PLOT = not self.PLOT
        GObject.timeout_add(42, self.plot)
        # GObject.idle_add(self.plot)

    def _change_density(self, event):
        self.logger.debug('SpingButton %s changed' % event)
        self.density = int(1.0 / self.density_spinentry.get_value())

    def _set_ranges(self, event):
        self.logger.debug('Button %s pressed' % event)
        try:
            vmin = float(self.vmin_entry.get_text())
        except:
            if self.zmin:
                vmin = self.zmin * 1.0
            else:
                return 1
        try:
            vmax = float(self.vmax_entry.get_text())
        except:
            vmax = self.zmax * 1.0
        if vmin >= vmax:
            self.logger.warning("Selected range is empty or illogical.")
            return 1
        else:
            self.logger.debug("Setting z range to: (%f, %f)" % (vmin, vmax))
        self.zlim = [vmin, vmax]
        # Update color bar
        self.reset = True

    @staticmethod
    def tstep(data, nsteps):
        return data['tstep'].value % nsteps

    @staticmethod
    def stepr(dt, xlimr, nmax):
        stepr = (xlimr[-1] - xlimr[0]) / dt / nmax
        return 1 if stepr < 1 else stepr

    def plot(self):
        self.tplot1 = self.tplot2 * 1
        self.tplot2 = self.data[self.tlabel].value * 1
        while Gtk.events_pending():
            Gtk.main_iteration()
        # Check if we are in a new cycle
        if self.cycle < self.tplot2 // self.nsteps:
            # New cycle (reset plot)
            self.cycle = self.tplot2 // self.nsteps
            t0 = (self.cycle * self.nsteps * self.dt + self.data[self.xlabel][0]) * self.tau
            # self.logger.debug("Cycle: %d. Actual time: %f" % (self.cycle, t0))
            self._resetplot()
        self.PLOT = self._plotdata()
        return self.PLOT

    def _wait(self):
        return False

    def _plotdata(self):
        tstep1 = self.tplot1 % self.nsteps
        tstep2 = self.tplot2 % self.nsteps
        # self.logger.debug("Tplot1: %d\tTplot2: %d" % (self.tplot1, self.tplot2))
        # self.logger.debug("Tstep1: %d\tTstep2: %d" % (tstep1, tstep2))
        if tstep2 == tstep1:  # When both tsteps are equal means the simulation is not running
            self.logger.warning("Time steps are equal, is the simulator running?")
            if self.counter <= self.tries:
                self.counter += 1
                self.logger.debug("Trying again...")
                return self.PLOT
            else:
                self.logger.warning("Stopping continuous plotting. Try again rerunning.")
                self.counter = 0
                return False
        self.counter = 0

        x = np.arange(self.tplot1, self.tplot2, 1) * self.dt * self.tau
        z = np.array(self.data[self.labels[1]][tstep1:tstep2]) / self.tau
        zmax = np.max(z)
        zmin = np.min(z)
        if self.zmax and self.zmin:
            if zmax > self.zmax:
                self.zmax = zmax * 1.0
            if zmin > self.zmin:
                self.zmin = zmin * 1.0
        else:
            self.zmax = zmax * 1.0
            self.zmin = zmin * 1.0

        # self.logger.debug("Shapes of x and z: (%s, %s)" % (np.shape(x), np.shape(z)))
        if self.reset:
            self.p = self.ax.pcolormesh(x[::self.density], self.y, z[::self.density].T, cmap=plt.get_cmap('gray'),
                                        vmin=self.zlim[0], vmax=self.zlim[1])
            if self.bar:
                self.bar.ax.cla()
                self.bar = plt.colorbar(self.p, cax=self.bar.ax)
            else:
                self.bar.ax.cla()
                self.bar = plt.colorbar(self.p, ax=self.ax)
            self.reset = False
            self.fig.canvas.draw()
            # self.ax.draw_artist(self.bar.ax)
        else:
            self.p = self.ax.pcolormesh(x[::self.density], self.y, z[::self.density].T, cmap=plt.get_cmap('gray'),
                                        vmin=self.zlim[0], vmax=self.zlim[1])

        self.ax.draw_artist(self.p)
        self.fig.canvas.blit(self.ax.bbox)
        return self.PLOT

    def _take_snapshot(self, event):
        self.logger.debug('Button %s pressed' % event)

        """ It changes the data of both axis taking the information from the shared memory object."""
        tstep = (self.data[self.tlabel].value + 1) % self.nsteps
        if tstep != 0:
            zdata1 = self.data[self.labels[1]][tstep:]
            zdata2 = self.data[self.labels[1]][0:tstep]
            zdata = np.concatenate((zdata1, zdata2))
        else:
            zdata = self.data[self.labels[1]][::]
        zdata /= self.tau
        self.zmin = np.min(zdata)
        self.zmax = np.max(zdata)
        xdata = (np.array(self.data[self.xlabel]) + self.data['temps'].value - self.data[self.xlabel][-1]) * self.tau
        xlim = [xdata[0], xdata[-1]]
        # xlim = np.array([self.data['t'][0], self.data['t'][-1]]) * self.tau
        ydata = np.linspace(-np.pi, np.pi, self.n + 1)
        p = self.ax.pcolormesh(xdata[::self.density], ydata, zdata[::self.density].T, cmap=plt.get_cmap('gray'),
                               vmin=self.zlim[0],
                               vmax=self.zlim[1])
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(self.ylim)
        if self.bar:
            self.bar = plt.colorbar(p, cax=self.bar.ax)
        else:
            self.bar = plt.colorbar(p, ax=self.ax)
        self.fig.tight_layout()
        self.canvas.draw()

    def _updatecursorposition(self, event):
        """When cursor inside plot, get position and print to status-bar"""
        if event.inaxes:
            x = event.xdata
            y = event.ydata
            self.statbar.push(1, ("Coordinates:" + " x= " + str(round(x, 3)) + "  y= " + str(round(y, 3))))


class RasterPlot(Gtk.Window):
    """ Gtk object containing a canvas plus some other widget, such as a toolbox."""

    def __init__(self, data, qin, qout, title='Raster', size=(800, 500), labels=('tfr', 'sp_re'), lims=None):
        """ Initialization requires a memory shared data object (dictionary). And some key values
            representing the variables to be plotted.
        """
        Gtk.Window.__init__(self, title=title)
        self.logger = logging.getLogger('gui.RasterPlot')

        self.window = self
        self.q_in = qin
        self.q_out = qout
        self.labels = labels
        self.lims = lims
        self.logger.debug("Limits are: %s" % self.lims[labels[1]])
        self.data = data

        self.set_default_size(*size)
        self.boxvertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.connect("delete-event", self._destroy)
        self.add(self.boxvertical)

        self.toolbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.boxvertical.pack_start(self.toolbox, False, False, 0)
        # In the upper part we locate a toolbox and Z range editor (all placed in a box)
        self.toolbar = Gtk.Toolbar()
        self.context = self.toolbar.get_style_context()
        self.context.add_class(Gtk.STYLE_CLASS_PRIMARY_TOOLBAR)
        self.toolbox.pack_start(self.toolbar, False, False, 0)

        self.snapshot_button = Gtk.ToolButton(icon_name='camera-photo')
        self.toolbar.insert(self.snapshot_button, 0)
        self.snapshot_button.connect("clicked", self._take_snapshot)

        # Modifying density of the plot

        self.urate = 100
        spin = Gtk.Adjustment(lower=1, upper=1000, step_increment=1, page_increment=10)
        self.urate_spinentry = Gtk.SpinButton(max_length=8, width_chars=8, xalign=1, adjustment=spin,
                                              tooltip_text='Change density of the plotting.', input_purpose='number',
                                              climb_rate=1, digits=0, numeric=True)
        self.urate_spinentry.set_value(self.urate)
        self.toolbox.pack_end(self.urate_spinentry, False, False, 0)
        self.urate_spinentry.connect("activate", self._change_urate)
        self.urate_spinentry.connect("value-changed", self._change_urate)

        self.box = Gtk.Box()
        self.boxvertical.pack_start(self.box, True, True, 0)

        # This can be put into a figure or a class ####################
        self.fig = plt.Figure(figsize=(10, 10), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        ###############################################################
        self.box.pack_start(self.canvas, True, True, 0)

        self.toolbar2 = NavigationToolbar(self.canvas, self)
        self.boxvertical.pack_start(self.toolbar2, False, True, 0)

        self.statbar = Gtk.Statusbar()
        self.boxvertical.pack_start(self.statbar, False, True, 0)

        self.fig.canvas.mpl_connect('motion_notify_event', self._updatecursorposition)

        # Figure plotting variables
        self.ylim = [-np.pi, np.pi]

        # Other variables
        self.PLOT = False
        self.reset = True

    # noinspection PyAttributeOutsideInit
    def initplot(self, **kwargs):
        # Color plot initialization
        self.n = kwargs['n']
        self.tau = kwargs['faketau']
        self.y = np.linspace(-np.pi, np.pi, self.n + 1)
        self.ax.set_xlabel(self.labels[0], fontsize='20')
        self.ax.set_ylabel(self.labels[1], fontsize='20')
        self.ax.set_yticks(np.arange(-np.pi, np.pi + np.pi / 2, np.pi / 2))
        y_ticklabels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
        self.ax.set_yticklabels(y_ticklabels, fontsize=14)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    # noinspection PyUnusedLocal
    def _destroy(self, *args):
        self.PLOT = False

    def _change_urate(self, event):
        self.logger.debug('SpinButton %s changed' % event)
        self.urate = int(self.urate_spinentry.get_value())
        self.q_in.put({'raster': {'rate': self.urate}})

    def _take_snapshot(self, event):
        self.logger.debug('Button %s pressed' % event)
        self.data.opts['raster']['update'] = True
        self.q_in.put({'raster': {'update': self.data.opts['raster']['update']}})
        raster_data = {}
        try:
            raster_data = self.q_out.get()
        except Queue.Empty:
            self.logger.debug("No data in the output queue.")

        if raster_data:
            self.ax.clear()
            for t in raster_data:
                y = raster_data[t]
                x = np.ones(len(raster_data[t]))*t*self.tau
                self.ax.plot(x, y, c='b', marker='.', markersize=1, linewidth=0)

        self.fig.tight_layout()
        self.canvas.draw()

    def _updatecursorposition(self, event):
        """When cursor inside plot, get position and print to status-bar"""
        if event.inaxes:
            x = event.xdata
            y = event.ydata
            self.statbar.push(1, ("Coordinates:" + " x= " + str(round(x, 3)) + "  y= " + str(round(y, 3))))


class RasterPlotC(Gtk.Window):
    """ Gtk object containing a canvas plus some other widget, such as a toolbox."""

    def __init__(self, data, multidata, qin, qout, title='Raster', size=(800, 500), labels=('tfr', 'sp_re'), lims=None):
        """ Initialization requires a memory shared data object (dictionary). And some key values
            representing the variables to be plotted.
        """
        Gtk.Window.__init__(self, title=title)
        self.logger = logging.getLogger('gui.RasterPlot')

        self.window = self
        self.q_in = qin
        self.q_out = qout
        self.labels = labels
        self.lims = lims
        self.logger.debug("Limits are: %s" % self.lims[labels[1]])
        self.data = data
        self.multidata = multidata

        self.set_default_size(*size)
        self.boxvertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.connect("delete-event", self._destroy)
        self.add(self.boxvertical)

        # Variables that must be assigned first:
        self.tplot1 = 0
        self.tplot2 = 0

        self.toolbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.boxvertical.pack_start(self.toolbox, False, False, 0)
        # In the upper part we locate a toolbox and Z range editor (all placed in a box)
        self.toolbar = Gtk.Toolbar()
        self.context = self.toolbar.get_style_context()
        self.context.add_class(Gtk.STYLE_CLASS_PRIMARY_TOOLBAR)
        self.toolbox.pack_start(self.toolbar, False, False, 0)

        self.refreshbutton = Gtk.ToolButton(icon_name='media-playback-start')
        self.toolbar.insert(self.refreshbutton, 0)
        self.refreshbutton.connect("clicked", self.run_dynamically)

        self.snapshot_button = Gtk.ToolButton(icon_name='camera-photo')
        self.toolbar.insert(self.snapshot_button, 0)
        self.snapshot_button.connect("clicked", self._take_snapshot)

        # Modifying rate of sampling
        self.urate = 100
        spin = Gtk.Adjustment(lower=1, upper=1000, step_increment=1, page_increment=10)
        self.urate_spinentry = Gtk.SpinButton(max_length=8, width_chars=8, xalign=1, adjustment=spin,
                                              tooltip_text='Change density of the plotting.', input_purpose='number',
                                              climb_rate=1, digits=0, numeric=True)
        self.urate_spinentry.set_value(self.urate)
        self.toolbox.pack_end(self.urate_spinentry, False, False, 0)
        self.urate_spinentry.connect("activate", self._change_urate)
        self.urate_spinentry.connect("value-changed", self._change_urate)

        self.box = Gtk.Box()
        self.boxvertical.pack_start(self.box, True, True, 0)

        # This can be put into a figure or a class ####################
        self.fig = plt.Figure(figsize=(10, 10), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        ###############################################################
        self.box.pack_start(self.canvas, True, True, 0)
        self.toolbar2 = NavigationToolbar(self.canvas, self)
        self.boxvertical.pack_start(self.toolbar2, False, True, 0)
        self.statbar = Gtk.Statusbar()
        self.boxvertical.pack_start(self.statbar, False, True, 0)
        self.fig.canvas.mpl_connect('motion_notify_event', self._updatecursorposition)

        # Other variables
        self.cycle = 0
        self.PLOT = False
        self.reset = True
        self.counter = 0
        self.tries = 5

    # noinspection PyAttributeOutsideInit
    def initplot(self, **kwargs):
        # Color plot initialization
        self.nsteps = self.data.nsteps
        self.n = kwargs['n']
        self.pop = self.data.opts['raster']['pop']
        if self.pop:
            self.ylim = [self.data.dNe * self.pop,self.data.dNe * (self.pop + 1)]
        else:
            self.ylim = [0, self.data.Ne]
        self.tau = kwargs['faketau']
        self.dt = self.data.dt
        self.pformat = {'c':'b', 'marker':'.', 'markersize':1, 'linewidth':0}
        p, = self.ax.plot(0, 0, **self.pformat)
        self.ax.set_xlabel(self.labels[0], fontsize='20')
        self.ax.set_ylabel(self.labels[1], fontsize='20')
        self.ax.set_xlim(np.array([self.data.vars['t'][0], self.data.vars['t'][-1]]) * self.tau)
        self.ax.set_ylim(self.ylim)
        self.fig.tight_layout()
        self.fig.canvas.draw()

        # We take the bounding boxes
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox.expanded(1.1, 1.1))

    def _resetplot(self):
        # Once a cycle is completed it erases the plot and starts again
        # Limits
        self.ax.clear()
        xlim = (self.multidata['cycle'].value * self.nsteps * self.dt + np.array([self.data.vars['t'][0],
                                                               self.data.vars['t'][-1]])) * self.tau
        self.ax.set_xlim(xlim)
        # Restore the region background
        self.fig.canvas.restore_region(self.bg)

    # noinspection PyUnusedLocal
    def _destroy(self, *args):
        self.PLOT = False

    def run_dynamically(self, button):
        self.logger.debug('Button %s pressed.' % button.get_name())
        self.PLOT = not self.PLOT
        self.data.opts['raster']['dynamic'] = not self.data.opts['raster']['dynamic']
        self.q_in.put({'raster': {'dynamic': self.data.opts['raster']['dynamic']}})
        if self.PLOT:
            self.snapshot_button.set_sensitive(False)
            self._resetplot()
        else:
            self.snapshot_button.set_sensitive(True)

        GObject.timeout_add(34, self.plot)
        # GObject.idle_add(self.plot)

    def _change_urate(self, event):
        self.logger.debug('SpinButton %s changed' % event)
        self.urate = int(self.urate_spinentry.get_value())
        self.q_in.put({'raster': {'rate': self.urate}})

    def _take_snapshot(self, event):
        self.logger.debug('Button %s pressed' % event)
        self.data.opts['raster']['update'] = True
        self.q_in.put({'raster': {'update': self.data.opts['raster']['update']}})
        raster_data = {}
        try:
            while not self.q_out.empty():
                raster_data = self.q_out.get()
        except Queue.Empty:
            self.logger.debug("No data in the output queue.")

        if raster_data:
            self.ax.clear()
            for t in raster_data:
                y = raster_data[t]
                if len(y) > 0:
                    x = np.ones(len(raster_data[t]))*t*self.tau
                    self.ax.plot(x, y, c='b', marker='.', markersize=1, linewidth=0)
                else:
                    raster_data.pop(t)

        self.fig.tight_layout()
        self.canvas.draw()

    def plot(self):
        while Gtk.events_pending():
            Gtk.main_iteration()
        # Check if we are in a new cycle
        if self.multidata['cycle'].value > self.cycle:
            self.cycle = self.multidata['cycle'].value
            # New cycle (reset plot)
            self._resetplot()
        self.PLOT = self._plotdata()
        return self.PLOT

    def _wait(self):
        return False

    def _plotdata(self):
        count = 0
        while not self.q_out.empty():
            if count > 20:
                break
            data = self.q_out.get_nowait()
            i = data.get('sp', False)
            if i is False:
                continue
            if len(i) > 0:
                t = data.get('t', False)
                if t is False:
                    continue
                t = data['t'] * self.tau * np.ones(len(i))
                try:
                    self.p, = self.ax.plot(t, i, **self.pformat)
                except ValueError:
                    self.logger.error("Something went wrong when plotting ...")
                    return self.PLOT
                self.ax.draw_artist(self.p)
            count += 1
        self.fig.canvas.blit(self.ax.bbox)
        return self.PLOT

    def _updatecursorposition(self, event):
        """When cursor inside plot, get position and print to status-bar"""
        if event.inaxes:
            x = event.xdata
            y = event.ydata
            self.statbar.push(1, ("Coordinates:" + " x= " + str(round(x, 3)) + "  y= " + str(round(y, 3))))


class DialogVar(Gtk.Dialog):
    def __init__(self, parent, model, **kwargs):
        Gtk.Dialog.__init__(self, "Select new variable", parent, 0,
                            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                             Gtk.STOCK_OK, Gtk.ResponseType.OK))

        self.logger = logging.getLogger('gui.DialogVar')
        self.set_default_size(150, 100)
        self.choice = ['t', 're']
        self.store = model

        box = self.get_content_area()
        box.set_border_width(15)
        labelx = Gtk.Label('New X (Angular) variable:')
        box.pack_start(labelx, True, True, padding=10)
        self.combo_x = Gtk.ComboBoxText.new()
        MainGui.update_combobox(self.combo_x, model)
        self.combo_x.connect("changed", self._on_plt_combo_x_changed)
        box.pack_start(self.combo_x, True, True, padding=10)
        labely = Gtk.Label('New Y (Polar) variable:')
        box.pack_start(labely, True, True, padding=10)
        self.combo_y = Gtk.ComboBoxText.new()
        MainGui.update_combobox(self.combo_y, model)
        self.combo_y.connect("changed", self._on_plt_combo_y_changed)
        box.pack_start(self.combo_y, True, True, padding=10)

        self.n = None
        self.pop_lbl = Gtk.Label('Population index (position): ')
        box.pack_start(self.pop_lbl, True, True, padding=10)
        pop_adjustment = Gtk.Adjustment(0, 100, 1, 10)
        self.pop_spin = Gtk.SpinButton(adjustment=pop_adjustment, numeric=True)
        self.pop_spin.connect("activate", self._on_pop_value_changed)
        self.pop_spin.connect("value-changed", self._on_pop_value_changed)
        box.pack_start(self.pop_spin, True, True, padding=10)
        self.pop_spin.hide()
        self.pop_lbl.hide()

        # Set default values
        target = 't'
        self.combo_x.set_active(0)
        i = 0
        while self.combo_x.get_active_text() != target:
            i += 1
            self.combo_x.set_active(i)
        if i == 0:
            self.combo_y.set_active(1)
        else:
            self.combo_y.set_active(0)
        self.show_all()

    def _on_pop_value_changed(self, spinbox):
        self.n = int(spinbox.get_value())

    def _on_plt_combo_x_changed(self, combo):
        """ Changing the combobox will set the variable tag to pass to the graphing class """
        name = combo.get_name()
        self.logger.debug('Element on %s modified' % name)
        # Let's get the name of the variable
        element = combo.get_active_text()
        self.choice[0] = element

    def _on_plt_combo_y_changed(self, combo):
        """ Changing the combobox will set the variable tag to pass to the graphing class """
        name = combo.get_name()
        self.logger.debug('Element on %s modified' % name)
        # Let's get the name of the variable
        element = combo.get_active_text()
        self.choice[1] = element

        # Show pop spin and label in case the variable is a matrix
        # Identify the rwo in the list store
        n = 0
        try:
            for row in self.store:
                if row[:][0] == element:
                    n = row[:][2]
        except IndexError:
            n = 1

        if n > 1:
            self.logger.debug("Showing population selection widgets.")
            adj = Gtk.Adjustment(lower=0, upper=n, value=int(n / 2), step_increment=1)
            self.pop_spin.set_adjustment(adj)
            self.pop_spin.show()
            self.pop_lbl.show()
            self.n = int(n / 2)
        else:
            self.pop_spin.hide()
            self.pop_lbl.hide()
            self.n = None


class SaveDialog(Gtk.Dialog):
    def __init__(self, parent, pvars, **kwargs):
        Gtk.Dialog.__init__(self, "Save data.", parent, 0,
                            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                             Gtk.STOCK_OK, Gtk.ResponseType.OK))
        self.logger = logging.getLogger('gui.DialogVar')
        self.set_default_size(150, 100)
        self.choice = 're'
        self.pvars = pvars
        var_type = type(pvars['t'])
        self._var_elements = MainGui.extract_tags(pvars, types=(var_type,))
        self.store = MainGui.update_tag_list(self._var_elements)
        self.all = False
        self.ic = False
        self.save_path = None
        self.ic_save_path = None

        try:
            self.save_path = kwargs['dir'] + '/result_' + '-'.join(now('_', '.'))
        except KeyError:
            pass

        box = self.get_content_area()
        box.set_border_width(15)

        # First line (label and combo)
        var_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        lbl = Gtk.Label('Variable to save:', xalign=0.0, yalign=0.5)
        var_box.pack_start(lbl, True, True, padding=10)
        self.combo_v = Gtk.ComboBoxText.new()
        MainGui.update_combobox(self.combo_v, self.store)
        self.combo_v.connect("changed", self._on_plt_combo_v_changed)
        var_box.pack_start(self.combo_v, True, True, padding=10)

        box.pack_start(var_box, True, True, padding=10)

        # Second line (label and switch)
        all_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        lbl2 = Gtk.Label('Save all variables:', xalign=0.0, yalign=0.5)
        all_box.pack_start(lbl2, True, True, padding=10)
        all_switch = Gtk.Switch(state=False)
        all_switch.set_hexpand(True)
        all_switch.set_halign(Gtk.Align.END)
        all_switch.set_name('all')
        all_switch.connect('notify::active', self._on_all_active)
        all_box.pack_start(all_switch, True, True, padding=10)

        box.pack_start(all_box, True, True, padding=10)

        # Third line (label and switch)
        ic_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        lbl3 = Gtk.Label('Save as initial conditions:', xalign=0.0, yalign=0.5)
        ic_box.pack_start(lbl3, True, True, padding=10)
        ic_switch = Gtk.Switch(state=False)
        ic_switch.set_hexpand(True)
        ic_switch.set_halign(Gtk.Align.END)
        ic_switch.set_name('ic')
        ic_switch.connect('notify::active', self._on_ic_active)
        ic_box.pack_start(ic_switch, True, True, padding=10)

        box.pack_start(ic_box, True, True, padding=10)

        # # Fourth line (label and switch)
        # ic_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        # lbl3 = Gtk.Label('Save as initial conditions:', xalign=0.0, yalign=0.5)
        # ic_box.pack_start(lbl3, True, True, padding=10)
        # ic_switch = Gtk.Switch(state=False)
        # ic_switch.set_hexpand(True)
        # ic_switch.set_halign(Gtk.Align.END)
        # ic_switch.set_name('ic')
        # ic_switch.connect('notify::active', self._on_ic_active)
        # ic_box.pack_start(ic_switch, True, True, padding=10)
        #
        # box.pack_start(ic_box, True, True, padding=10)

        # Fifth line (button, set path)
        self.set_path = Gtk.Button(label='Select saving path')
        self.set_path.connect("clicked", self._path_dialog)
        box.pack_start(self.set_path, False, False, padding=10)

        self.show_all()

    def _on_plt_combo_v_changed(self, combo):
        """ Changing the combobox will set the variable tag to pass to the graphing class """
        name = combo.get_name()
        self.logger.debug('Element on %s modified' % name)
        # Let's get the name of the variable
        element = combo.get_active_text()
        self.choice = element

    def _on_all_active(self, switch, state):
        name = switch.get_name()
        self.logger.debug("Switch %s: %s" % (name, state))
        self.all = switch.get_active()
        if self.all:
            self.combo_v.set_sensitive(False)
        else:
            self.combo_v.set_sensitive(True)
        return 0

    def _on_ic_active(self, switch, state):
        name = switch.get_name()
        self.ic = switch.get_active()
        if self.ic:
            self.set_path.set_sensitive(False)
        else:
            self.set_path.set_sensitive(True)
        return 0

    def _path_dialog(self, button):
        dialog = Gtk.FileChooserDialog("Please choose a file", self, Gtk.FileChooserAction.SAVE,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        add_filters(dialog)
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_current_name(self.save_path)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.save_path = dialog.get_filename()
            print("File selected: " + dialog.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            pass

        dialog.destroy()
