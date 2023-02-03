import numpy as np
import matplotlib.patches as pat


def plotFrame(axes, origin, xyz_sign=[1, 1, 1], color_seq=['r', 'g', 'b'], percent_figure=.2):
    '''
    axes: Axes, matplotlib
    origin: 1D-array, len=2, float
    xyz_sign: 1D-array, len=3, +/- 1
    color_seq: 1D-array, len=3, colors
    percent_figure: float>0

    This function prints a xyz frame, scaled on axes's xlim and ylim
    Default is an rgb xyz frame with x-y axes equal to those of axes 

    axes are the axes of the plot
    origin is the origin of the frame
    xyz_sign are signs applied to figure axes
    color_seq are colors applied to axes (in cannonical xyz order)
    percent_figure is the size of the arrows of the frame, divided by the size of the figure
    '''

    # compute arrow length from axes.x_lim and axes.y_lim
    min_delta = min(axes.get_xlim()[1] - axes.get_xlim()
                    [0], axes.get_ylim()[1] - axes.get_ylim()[0])
    arrow_length = min_delta*percent_figure
    # arrow_width = .005 + (percent_figure-.2)/60  # hand tuned parameter

    # Draw x-vector
    x_vector = xyz_sign[0]*np.array([arrow_length, 0])
    axes.arrow(*origin, *x_vector, color=color_seq[0])

    # Draw y-vector
    y_vector = xyz_sign[1]*np.array([0, arrow_length])
    axes.arrow(*origin, *y_vector, color=color_seq[1])

    # Draw z-vector
    if(xyz_sign[2] == 1):
        # pointing "toward the eye"
        radius_z_vector = arrow_length/30

        axes.add_patch(pat.Circle(tuple(origin), radius_z_vector,
                       color=color_seq[2], fill=True))
        axes.add_patch(pat.Circle(tuple(origin), 2.5 *
                       radius_z_vector, color=color_seq[2], fill=False))

    else:
        # draw cross base of vector
        z_cross_size = arrow_length/18

        same_sign_branch = np.stack(
            [origin+z_cross_size, origin-z_cross_size]).T

        diff_sign_slack = np.array([z_cross_size, -z_cross_size])
        diff_sign_branch = np.stack(
            [origin+diff_sign_slack, origin-diff_sign_slack]).T

        cross_width = 4+7*(percent_figure-.2)
        axes.plot(same_sign_branch[0], same_sign_branch[1],
                  lw=cross_width, c=color_seq[2])
        axes.plot(diff_sign_branch[0], diff_sign_branch[1],
                  lw=cross_width, c=color_seq[2])
        axes.add_patch(pat.Circle(tuple(origin), 2*z_cross_size,
                       color=color_seq[2], fill=False))
