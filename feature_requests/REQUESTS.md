# BVR Motion Viewer Feature Request Documentation

This document tracks feature requests, suggestions and bug fixes for the BVR Motion Viewer, from when they are requested to when they are built into the code.

**How to use it**

- Add each new request as a new row at the bottom of the table, with the date it was requested (d/mm/yyyy), the status `Request`, and a new number (#).
- Change the status to `Implementing` while it is being worked on, and to `Completed` once it is in the code.
- When a request is completed, the **Notes** column records how it was implemented (what changed, where, and the commit).
- To keep a line break inside a cell, use `<br>`. Every row must stay on one line.

**Status values:** `Request` · `Implementing` · `Completed`

---

| # | Date | Status | Request | Notes |
|---|---|---|---|---|
| 1 | 9/11/2026 | Request | All options that have a slider input should have the capacity for the number that the slider is for to be changed with text (i.e. text box entry for the input alongside the slider). | |
| 2 | 9/11/2026 | Request | Navigator plots should default to legend labels of 'Angle X/Y/Z' and 'Translation X/Y/Z'. | |
| 3 | 9/11/2026 | Request | Navigator plots should have the option to plot both bone and joint motion, with joint motion only an option where a model is active. | |
| 4 | 9/11/2026 | Request | The navigator tab trees should work in the following way:<br>(1) once a bone is loaded in, it should have its name listed in the tree with a separate tree underneath this where the 'Axes' and 'Tracking' options are listed, where these options remain greyed out or inactive until something is loaded for them;<br>(2) you should be able to right-click on the axes and tracking tree options and have a menu come up to load these;<br>(3) once loaded these parts of the tree should become 'active' and be bolded or change to a relevant colour;<br>(4) there should be an option to right click on the active aspect and give the menu to either load a different file or refresh the current file (i.e. if the file has been saved over and you refresh, the new axes or tracking will replace this). | |
| 5 | 9/11/2026 | Request | Given that Autoscoper doesn't export timestamps in its tracking exports, there needs to be an option for dealing with tracking files of different lengths. Autoscoper inserts NaNs into non-tracked frames, but currently we rely on users inputting tracking files where the frames correspond to one another. There may need to be an option to warn on files where different frame lengths are loaded. | |
| 6 | 9/11/2026 | Request | Autoscoper can export tracking data in different ways, such as by row or column, or in centimetres or millimetres. Perhaps this can be automatically identified based on the shape or structure of the file input, however it may be safer to prompt for the same inputs that Autoscoper does when you save/load a tracking file. | |
| 7 | 9/11/2026 | Request | Currently when opening a 'Visualiser' or 'Model Builder' tab, the entire window closes down and a new window with the tabs opens. The preferable behaviour is for the same window to remain, have the tab open in that window, but then still have the capacity to open up different tab types in this same main window. | |
| 8 | 9/11/2026 | Request | In the Visualiser window, we should have an option to have multiple window tabs within this or have a 3-window-like set-up within this so that a person could view: (i) the 3D viewer of the tracked bones; (ii) the other two windows correspond to 2D camera images of the X-ray with the tracked bones overlaid (with an option for these overlays to be point clouds or a 2D overlay similar to what Autoscoper looks like). The colouring of the overlays or point clouds in these camera windows should match the colour of the bone loaded in the 3D view. | |
| | | | | |
| | | | | |
| | | | | |
