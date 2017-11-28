from PsiMarginal import Psi


def psi_handler(psi_queue=None, work_done_queue=None, psi_options=dict, verbose=False):
    """
    Worker function for the psi marginal calculation.
    Creates a dict of psi marginal objects, calculates the currently estimated
    most optimal stimulus value to test, and updates the objects when a
    response is given.
    :param psi_queue: : queue object. Takes a dict with keys:
        *'condition'* to create psi object and/or get the current estimated
        optimal stimulus value to test
        *'response'* (optional) to update the psi marginal object with the
        given response
        Input 'STOP' to end the process.
    :param work_done_queue: queue object. Sends a dict to the main process with
     'condition' and 'stimValue'
    :param psi_options: dict, must at least contain 'stimRange': the search range for stimulus values.
    :param verbose: optional, prints the current activity. False by default.
    For further options, see the documentation of PsiMarginal.
    """

    _psiobjects = {}

    while True:
        data = psi_queue.get()
        if data == 'STOP':
            if verbose: print('PSI_HANDLER: end')
            break
        else:
            cond_id = data['condition']
            obj = _psiobjects.get(cond_id, None)

            # instantiate psi object if it doesn't exist yet
            if obj is None:
                if verbose: print('PSI_HANDLER: creating psi object with ID {}'.format(cond_id))
                try:
                    _psiobjects[cond_id] = Psi(psi_options['stimRange'])
                except:
                    print('PSI_HANDLER: psi_options dict in must contain a stimulus search range with key \'stimRange\'')
                    break
                obj = _psiobjects[cond_id]

            # check for response
            response = data.get('response', None)

            # if no response given, send current rod angle to the main loop
            if response is None:
                work_done_queue.put({'condition': cond_id, 'stimValue': obj.xCurrent})
            # if response is given, update the psi object
            else:
                obj.addData(response)
