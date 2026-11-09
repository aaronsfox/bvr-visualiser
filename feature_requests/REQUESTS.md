# BVR Motion Viewer Feature Request Documentation

This document provides a  timeine of  feature requests and adjustments built into the BVR motion viewer platform. Separate sections relate to rough timestamps as to when features are requested/suggested. Checkboxes are used to signify once the feature is addressed.

---

## November 9th, 2026

- [ ]  All options that have a slider input should have the capacity for the number that the slider is for to be changed with text (i.e. text box entry for the input alongside the slider).

- [ ]  Navigator plots should default to legend labels of 'Angle X/Y/Z' and Translation X/Y/Z'

- [ ]  Navigator plots should have the option to plot both bone and joint motion, with joint motion only an option where a model is active

- [ ]  The navigator tab trees should work in the following way: (1) once a bone is loaded in, it should have it's name listed in the tree with a separate tree underneath this where the 'Axes' and 'Tracking' options are listed, where these options remain greyed out or inactive until something is loaded for them; (2) you should be able to right-click on the axes and tracking tree options and have a menu come up to load these; (3) once loaded these parts of the tree should become 'active' and be bolded or change to a relevant colour; (4) there should be an option to right click on the active aspect and give the menu to either load a different file or refresh the current file (i.e. if the file has been saved over and you refresh - the new axes or tracking will replace this)

- [ ]  Given that Autoscoper doesn't export timestamps in it's tracking exports, there needs to be an option for dealing with tracking files of different lengths. Autoscoper inserts NaN's into non-tracked frames, but currently we rely on users inputting tracking files where the frames correspond to one another. There may need to be an option to warn on files where different frame lengths are loaded.  

- [ ]  Autoscoper can export tracking data in different ways - such as by row or column, or be centimetres or millimetres. Perhaps this can be automatically identified based on the shape or structure of the file input, however it may be safer to prompt for the same inputs that Autoscoper does when you save/load a tracking file. 

- [ ]  Currently when opening on a 'visualiser' or 'modle builder' tab, the entire window closes down and a new window with the tabs opens.  The more preferable behaviour for the same window to remain, have the tab open in that window, but  then still have the capacity to open up different tab types in this same main window. 

- [ ]  In the visualiser window, we should have an optionto have multiple window tabs within this or have a 3 window-like set-up within this so that a person could view: (i) the 3D viewer of the tracked bones; (ii) the other two windows correspond to 2D camera images of the X-ray with the tracked bones overlaid (with an option for these overlays to be point clouds or a 2D overlay similar to what autoscoper looks like). The colouring of the overlays or point clouds in these camera windows should match to the colour of the bone loaded in the 3D view.

---
