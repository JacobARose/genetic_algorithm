

import os
__PACKAGE_ROOT__ = os.path.dirname(__file__)
__PACKAGE_RESOURCES__ = os.path.abspath(os.path.join(__PACKAGE_ROOT__,'..', 'resources'))
__PACKAGE_TEST_DIR__ = os.path.abspath(os.path.join(__PACKAGE_ROOT__,'..', 'tests'))



from . import utils
from . import models
from . import datasets
from . import generation
from . import organism
from . import stateful
from . import chromosome