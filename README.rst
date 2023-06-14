|GitHub version|  |Licence GPLv3| |Python version| |OS|


.. |GitHub version| image:: https://img.shields.io/badge/version-0.0.dev0-yellow.svg
   :target: https://github.com/mikecokina/photoshop

.. |Python version| image:: https://img.shields.io/badge/python-3.6|3.7|3.8|3.9-orange.svg
   :target: https://github.com/mikecokina/photoshop

.. |Licence GPLv3| image:: https://img.shields.io/badge/license-GNU/GPLv3-blue.svg
   :target: https://github.com/mikecokina/photoshop

.. |OS| image:: https://img.shields.io/badge/os-Linux|Windows-magenta.svg
   :target: https://github.com/mikecokina/photoshop


Photoshop algorithms Python Port
================================

- Brightnes and contrast
    - Photoshop like smooth implementation
    - Legacy implementation
    - Automatic contrast implementation

- Filters
    - Distort
        - Displacement
            - doesn't allow `x` and `y` separately, currenlty just single parameter `strength` is supported
            - doesn't allow edge behavior definition

- Blending Options
    - Normal Blending
        - supports conditional blending (Blend If) for background image (doesn't support foreground image condition)
        - doesn't support codnitional blending for overlapping intervals
        - supports only condition based on gray scale of background image
        - doesn't support opcaity option
