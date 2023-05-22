
class IdentifierUnderspecifiedException(Exception):
    pass

class NoRunException (Exception):
    pass

class SomebodyElseWasFasterException(Exception):
    pass

class InValidRunParameterException(Exception):
    pass

class UnknownDatasetException(Exception):
    pass

class UnableToMakeRunPersistent(Exception):
    pass

def parse_identifier(identifier : str):
    """
    Given a str of either one number (attack_id) or two number separated by '-', it returns a tuples (attack_id,gs_id). This allows to
    load by attack_id if there is only one gs_id for this attack but also to specify the gs_id if required.
    :param identifier:
    :type identifier: str
    """
    if "-" in identifier:
        s = identifier.split("-")
        if len(s) != 2:
            raise Exception(f"identifier has to be either only a number,'<attack_id>' or a string '<attack_id>-<gs_id>'. It is {identifier}")
        attack_id = int(s[0])
        gs_id = int(s[1])
    else:
        attack_id = int(identifier)
        gs_id = None

    return (attack_id, gs_id)

from .Experiment import *
from .Base import *
from .Run import *
