from os.path import dirname, abspath, isdir
import numpy as np
from os import mkdir
import multiprocessing
from collections import OrderedDict
from psychopy import visual, core, event
from random import shuffle
from psi_handler import psi_handler
from stateMachine import StateMachine
import time


class RunExp(object):
    """
    All stimuli, methods and settings needed to run the experiment.
    """
    def __init__(self):

        # experiment information
        self.sj_id = raw_input('Subject ID: ').upper()
        self.paradigm = 'OK'
        self.ok_condition = raw_input('Condition (CW, CCW or NONE): ').upper()
        self.frame_angles = [-45, -33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75,
                             'noframe']
        self.ntrials = 60
        self.break_after_trials = 108
        self.break_trials = range(self.break_after_trials,
                                  len(self.frame_angles) * self.ntrials,
                                  self.break_after_trials)
        self.optokinetic_velocities = {'CW': 30.0, 'CCW': -30.0, 'NONE': 0.0}
        self.dots_rotation = self.optokinetic_velocities[self.ok_condition]

        # window and display settings
        self.win = visual.Window(size=[1920, 1080], color=(-1, -1, -1),
                                 monitor='OLED', winType='pygame', units='pix',
                                 fullscr=True)
        self.winWidth, self.winHeight = self.win.size
        self.framerate = self.win.fps()
        self.frame_dur = 1.0 / self.framerate
        self.mouse = event.Mouse(visible=False, win=self.win)

        # initiate queues and process for psi marginal adaptive sampling procedure
        self.psi_options = {'stimRange': np.arange(-30, 30, 0.5)}
        self.psi_queue = multiprocessing.Queue()
        self.work_done_queue = multiprocessing.Queue()
        self.psiProcess = multiprocessing.Process(target=psi_handler,
                                                  args=(self.psi_queue,
                                                        self.work_done_queue,
                                                        self.psi_options),
                                                  kwargs={'verbose': True})

        # experiment phase durations
        self.durations = {
            'start': False,
            'habituation': 30.0,
            'init_trial': False,
            'iti': 0.25,
            'pre_probe': 0.25,
            'probe': 0.033,
            'response': 2.0
        }

        # initiate data storage and stimuli
        self.data = {}
        self.trials = self.__trial_list__()
        self.create_stimuli()
        self.datafile = self.__data_file__()

    def __data_file__(self):
        """
        Creates a data folder and file for each subject.
        :return: absolute path of data file
        """

        self.rootfolder = dirname(abspath('__file__'))
        self.datafolder = '%s/data' % self.rootfolder
        self.subfolder = '{}/{}'.format(self.datafolder, self.sj_id)
        if not isdir(self.datafolder):
            mkdir(self.datafolder)
        if not isdir(self.subfolder):
            mkdir(self.subfolder)

        # data file
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.datafile = '{}/{}_{}_{}_{}.txt'.format(self.subfolder,
                                                    self.sj_id,
                                                    self.paradigm,
                                                    self.ok_condition,
                                                    self.timestr)

        # write column header to datafile
        with open(self.datafile, 'a') as current_file:
            current_file.write('trialNr, trialOnset, dotsRotation, frameAngle,'
                               'rodAngle, response\n')

        return self.datafile

    def __trial_list__(self):
        self.triallist = list(self.frame_angles) * self.ntrials
        shuffle(self.triallist)
        return self.triallist

    def create_stimuli(self):
        self.stimuli = OrderedDict()

        self.stimuli['dotsBackground'] = visual.DotStim(
            win=self.win,
            nDots=500,
            coherence=1,
            fieldSize=1060,
            fieldShape='circle',
            dotSize=2.0,
            dotLife=100.0,
            speed=0,
            color=(-0.8, -0.8, -0.8),
            signalDots='same',
            noiseDots='direction',
            units='pix')
        self.stimuli['dotsBackground'].ori = 0

        self.stimuli['circlePatch'] = visual.Circle(
            win=self.win,
            radius=240.0,
            pos=(0, 0),
            lineWidth=1,
            lineColor=(-1, -1, -1),
            fillColor=(-1, -1, -1),
            units='pix')

        self.stimuli['rodStim'] = visual.Line(
            win=self.win,
            start=(0, -100),
            end=(0, 100),
            lineWidth=5,
            lineColor=(-0.84, -0.84, -0.84))

        self.stimuli['squareFrame'] = visual.Rect(
            win=self.win,
            width=300.0,
            height=300.0,
            pos=(0, 0),
            lineWidth=5,
            lineColor=(-0.84, -0.84, -0.84),
            fillColor=None,
            ori=0.0,
            units='pix')

        # stimulus triggers
        self.triggers = {}
        for stim in self.stimuli:
            self.triggers[stim] = False

    @staticmethod
    def rotate_stimulus(stimulus, start_time, angular_velocity):
        """
        Rotate an existing stimulus at *angular_velocity* degrees per second,
        starting at start_time.
        """
        stimulus.ori = angular_velocity * round(time.time() - start_time,
                                                ndigits=3)

    def display_stimuli(self):
        for stim in self.stimuli:
            if self.triggers[stim]:
                if stim == 'dotsBackground':
                    self.rotate_stimulus(self.stimuli[stim], self.start_time,
                                         self.dots_rotation)
                self.stimuli[stim].draw()
        self.win.flip()

    def check_response(self):
        """
        Check for participant response (key press)
        """
        self.key_response = event.getKeys(
            keyList=['left', 'right', 'space', 'escape'])
        self.state_change = None
        if self.key_response:
            if 'left' in self.key_response:
                self.data['response'] = False
                self.state_change = 'init_trial'
            elif 'right' in self.key_response:
                self.data['response'] = True
                self.state_change = 'init_trial'
            elif 'space' in self.key_response:
                self.state_change = 'pause'
            elif 'escape' in self.key_response:
                self.state_change = 'end'
        return self.state_change

    def check_keys(self):
        """
        Check for key presses
        """
        self.key_presses = event.getKeys(keyList=['space', 'escape'])
        event.clearEvents(eventType='mouse')
        if self.key_presses:
            if 'space' in self.key_presses:
                self.state_change = 'pause'
            elif 'escape' in self.key_presses:
                self.state_change = 'end'
        else:
            self.state_change = None
        return self.state_change

    def pause_screen(self, current_trial):
        pauseScreen = visual.Rect(
            win=self.win,
            width=self.winWidth,
            height=self.winHeight,
            lineColor=(0, 0, 0),
            fillColor=(0, 0, 0))

        pauseText = visual.TextStim(
            win=self.win,
            text='PAUSE  trial {}/{} Press space to continue'.format(
                current_trial, len(self.triallist)),
            pos=(0.0, 0.0),
            color=(-1, -1, 0.6),
            units="pix",
            height=40)

        pauseScreen.draw()
        pauseText.draw()
        self.win.flip()

    def save_data(self):
        self.formatted_data = '{}, {}, {}, {}, {}, {}\n'.format(
            self.data['trialNr'], self.data['trialOnset'], self.ok_condition,
            self.data['frameAngle'], self.data['rodAngle'],
            self.data['response'])
        with open(self.datafile, 'a') as current_file:
            current_file.write(self.formatted_data)

    def quit_exp(self):
        """
        stop psi marginal process, close window and quit experiment
        """
        self.psi_queue.put('STOP')
        self.psiProcess.join()
        self.win.close()
        core.quit()


class RunStates(RunExp):
    """
    Defines and runs the experiment phases as states, with the experiment class
    RunExp as an argument.
    """
    def __init__(self):
        """
        Initialise finite state machine
        """
        RunExp.__init__(self)

        self.fsm = StateMachine()
        self.fsm.add_state('start', self.start_handler, end_state=False)
        self.fsm.add_state('habituation', self.habituation_handler,
                           end_state=False)
        self.fsm.add_state('pause', self.pause_handler, end_state=False)
        self.fsm.add_state('init_trial', self.init_trial_handler,
                           end_state=False)
        self.fsm.add_state('iti', self.iti_handler, end_state=False)
        self.fsm.add_state('pre_probe', self.pre_probe_handler,
                           end_state=False)
        self.fsm.add_state('probe', self.probe_handler, end_state=False)
        self.fsm.add_state('response', self.response_handler, end_state=False)
        self.fsm.add_state('end', self.end_handler, end_state=True)
        self.fsm.set_start('start')
        self.go_next = False

    def start_handler(self):
        """
        start psi process and instantiate all psi marginal objects
        """
        self.psiProcess.start()
        for frame_ang in self.frame_angles:
            self.psi_queue.put({'condition': frame_ang})
            self.work_done_queue.get()

        # triggers for timers
        self.timer_triggers = {}
        self.statenames = ['start', 'habituation', 'init_trial', 'iti',
                           'pre_probe', 'probe', 'response']
        for state in self.statenames:
            self.timer_triggers[state] = True

        self.start_time = time.time()
        self.trial_count = 0
        self.newState = 'habituation'
        self.go_next = True
        return (self.newState, self.go_next)

    def pause_handler(self):
        for stim in self.triggers:
            self.triggers[stim] = False
        self.pause_screen(self.trial_count)
        event.waitKeys(maxWait=float('inf'), keyList=['space'])

        # restart habituation and replay interrupted trial
        self.trial_count -= 1
        self.timer_triggers['habituation'] = True
        self.newState = 'habituation'
        self.go_next = True
        return (self.newState, self.go_next)

    def habituation_handler(self):
        if self.timer_triggers['habituation']:
            self.habituation_timer = time.time()
            self.timer_triggers['habituation'] = False
        self.triggers['dotsBackground'] = True
        self.triggers['circlePatch'] = True
        self.display_stimuli()
        self.newState = self.check_keys()
        if self.newState == 'pause':
            self.go_next = True
        elif self.newState == 'end':
            self.go_next = True
        else:
            self.newState = 'init_trial'
            if (time.time() - self.habituation_timer) > self.durations[
                'habituation']:
                self.go_next = True
            else:
                self.go_next = False
        return (self.newState, self.go_next)

    def init_trial_handler(self):
        self.data = {}
        self.data['trialNr'] = self.trial_count
        self.trial_count += 1

        # check if this is a break trial
        if self.trial_count in self.break_trials:
            # increment count because this trial doesn't need to be repeated
            self.trial_count += 1
            self.newState = 'pause'
            self.go_next = True
            return (self.newState, self.go_next)

        # trial settings
        self.rodAngle = None
        self.data['trialOnset'] = time.time()
        self.data['frameAngle'] = self.trials[self.trial_count]
        self.psi_queue.put({'condition': self.data['frameAngle']})
        if self.data['frameAngle'] != 'noframe':
            self.stimuli['squareFrame'].ori = self.data['frameAngle']
        self.triggers['dotsBackground'] = True
        self.triggers['circlePatch'] = True
        self.display_stimuli()

        # reset timer triggers
        for state in self.statenames:
            self.timer_triggers[state] = True
        self.newState = 'iti'
        self.go_next = True
        return (self.newState, self.go_next)

    def iti_handler(self):
        if self.timer_triggers['iti']:
            self.iti_timer = time.time()
            self.timer_triggers['iti'] = False

        # attempt to retrieve optimal probe rod angle from psi marginal procedure
        if self.rodAngle is None:
            try:
                self.rodAngle = self.work_done_queue.get(block=False)[
                    'stimValue']
            except:
                pass
            else:
                self.data['rodAngle'] = self.rodAngle
                self.stimuli['rodStim'].ori = self.rodAngle

        self.triggers['dotsBackground'] = True
        self.triggers['circlePatch'] = True
        self.display_stimuli()

        self.newState = self.check_keys()
        if self.newState == 'pause':
            self.go_next = True
        elif self.newState == 'end':
            self.go_next = True
        else:
            self.newState = 'pre_probe'
            if (time.time() - self.iti_timer) > self.durations[
                'iti'] and self.rodAngle is not None:
                self.go_next = True
            else:
                self.go_next = False
        return (self.newState, self.go_next)

    def pre_probe_handler(self):
        if self.timer_triggers['pre_probe']:
            self.pre_probe_timer = time.time()
            self.timer_triggers['pre_probe'] = False
        self.triggers['dotsBackground'] = True
        self.triggers['circlePatch'] = True
        if self.data['frameAngle'] != 'noframe':
            self.triggers['squareFrame'] = True
        self.display_stimuli()
        self.newState = self.check_keys()
        if self.newState == 'pause':
            self.go_next = True
        elif self.newState == 'end':
            self.go_next = True
        else:
            self.newState = 'probe'
            if (time.time() - self.pre_probe_timer) > self.durations[
                'pre_probe']:
                self.go_next = True
            else:
                self.go_next = False
        return (self.newState, self.go_next)

    def probe_handler(self):
        if self.timer_triggers['probe']:
            self.probe_timer = time.time()
            self.timer_triggers['probe'] = False
        self.triggers['dotsBackground'] = True
        self.triggers['circlePatch'] = True
        if self.data['frameAngle'] != 'noframe':
            self.triggers['squareFrame'] = True
        self.triggers['rodStim'] = True
        self.display_stimuli()
        self.newState = self.check_keys()
        if self.newState == 'pause':
            self.go_next = True
        elif self.newState == 'end':
            self.go_next = True
        else:
            self.newState = 'response'
            if (time.time() - self.probe_timer) > self.durations['probe']:
                self.go_next = True
            else:
                self.go_next = False
        return (self.newState, self.go_next)

    def response_handler(self):
        if self.timer_triggers['response']:
            self.response_timer = time.time()
            self.timer_triggers['response'] = False
        self.triggers['dotsBackground'] = True
        self.triggers['circlePatch'] = True
        if self.data['frameAngle'] != 'noframe':
            self.triggers['squareFrame'] = True
        self.triggers['rodStim'] = False
        self.display_stimuli()

        # if response is given, save data and go to next trial
        self.newState = self.check_response()
        if self.newState == 'init_trial':
            self.triggers['squareFrame'] = False
            self.psi_queue.put({'condition': self.data['frameAngle'],
                                'response': self.data['response']})
            self.save_data()
            if self.trial_count == (len(self.trials) - 1):
                self.newState = 'end'
            self.go_next = True
            return self.newState, self.go_next
        elif self.newState == 'pause':
            self.go_next = True
        elif self.newState == 'end':
            self.go_next = True
        elif (time.time() - self.response_timer) > self.durations['response']:
            # when a trial times out, rerun that trial
            self.trial_count -= 1
            self.triggers['squareFrame'] = False
            self.newState = 'init_trial'
            self.go_next = True
        else:
            self.newState = 'init_trial'
            self.go_next = False
        return (self.newState, self.go_next)

    def end_handler(self):
        self.newState = None
        self.go_next = True
        return (self.newState, self.go_next)


if __name__ == '__main__':
    experiment = RunStates()
    experiment.fsm.run()
    experiment.quit_exp()
