"""
Test for schedule
"""

import logging
import unittest

from winternlc.get_corrections import check_for_files

logger = logging.getLogger(__name__)


class TestDownload(unittest.TestCase):
    """
    Class for testing
    """

    def test_download(self):
        """
        Test nlc correction on test image

        :return: None
        """
        logger.info("Testing nlc")

        check_for_files()
