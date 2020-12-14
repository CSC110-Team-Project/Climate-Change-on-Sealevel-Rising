"""CSC110 Final Project 2020: main.py

Copyright and Usage Information
===============================
This file is part of the CSC110 final project: Data Analysis on Rising Sea Level,
developed by Charlie Guo, Owen Zhang, Terry Tu, Vim Du.
This file is provided solely for the course evaluation purposes of CSC110 at University of Toronto St. George campus.
All forms of distribution of this code, whether as given or with any changes, are strictly prohibited.
The code may have referred to sources beyond the course materials, which are all cited properly in project report.
For more information on copyright for this project, please contact any of the group members.

This file is Copyright (c) 2020 Charlie Guo, Owen Zhang, Terry Tu and Vim Du.
"""
from visualization import *
from CNNprocess import *
from NNprocess import *
from preprocess import Preprocess


def perform_process_data() -> pd.DataFrame:
    """Perform the processing of data"""
    processor = Preprocess()
    sealvl = processor.sea_level_process('sea_levels_2015.csv')
    seaice = processor.seaice_process('seaice.csv')
    temp = processor.temperature_process('GlobalTemperatures.csv')
    data = processor.merge_data(seaice, temp, sealvl)
    return data


def perform_cnn(period: int, data: pd.DataFrame) -> \
        Tuple[CNNProcess, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform CNN model's computation"""
    cnnp = CNNProcess(period, 5)
    cnnp.load_data(data)
    cnnp.train_model()
    cnnp.test_data()
    cnn_test, cnn_test_pred = cnnp.predict_result()
    cnn_pred, cnn_act = cnnp.pred_validation(period)
    return (cnnp, cnn_test, cnn_test_pred, cnn_pred, cnn_act)


def perform_nn(period: int, data: pd.DataFrame) -> Tuple[NNProcess, np.ndarray, np.ndarray]:
    """Perform NN model's computation"""
    # Perform NN model's calculation
    nnp = NNProcess(period)
    nnp.load_data(data)
    nnp.train_model()
    nnp.test_data()
    nn_test, nn_test_pred = nnp.predict()
    return (nnp, nn_test, nn_test_pred)


def write_file_local(period, CNN_act: np.ndarray, CNN_pred: np.ndarray) -> None:
    """Write the predicted result to the csv file"""
    s_year, s_month = calculate_start_date(2013, 12, period)  # the start year and start month
    cnn_total = np.append(CNN_act[-period:], CNN_pred)  # The interval we consider is [2013.12-period, 2013.12 + period)
    write_file('CNN_output.csv', s_year, s_month, len(cnn_total), cnn_total)


def visualization_1(CNNP: CNNProcess, test_pred: np.ndarray, test: np.ndarray) -> None:
    """Perform the comparison graph of CNN model's test prediction and actual data in the predicted period"""
    CNNP.visualization(test_pred, test)


def visualization_2(NNP: NNProcess) -> None:
    """Perform the comparison of NN model's prediction and actual data in the predicted period."""
    nn_test, nn_test_pred = NNP.predict()
    NNP.visualization(nn_test, nn_test_pred)


def visualization_3(data: pd.DataFrame, period: int) -> None:
    """Perform the animated graph to illustrate the predicted result of CNN model."""
    cnn = pd.read_csv('CNN_output.csv')
    cnn_animated = merge_date(cnn)
    actual = merge_date(data)[-period:]
    graph = plot_animated_graph(cnn_animated, actual, 'Date', 'GMSL', period)
    graph.show()


def visualization_4(act: np.ndarray, pred: np.ndarray) -> None:
    """Perform the 3D scatter plot of the predicted result of CNN Model.
    Note that all the data after 1978-10 will be displayed. """
    cnn_full = np.append(act, pred)
    write_file('CNN_full_output.csv', 1978, 10, len(cnn_full), cnn_full)
    full_cnn = pd.read_csv('CNN_full_output.csv')
    graph_2 = plot_3d_scatter_plot(full_cnn, 'Year', 'Month', 'GMSL', 'Year')
    graph_2.show()


if __name__ == "__main__":
    import pygame
    import pygame_gui

    pygame.init()

    pygame.display.set_caption('GMSL Prediction')
    surface = pygame.display.set_mode((800, 1000))
    background = pygame.Surface((800, 1000))
    background.fill(pygame.Color('#000000'))

    manager = pygame_gui.UIManager((800, 600))

    process_data = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 150), (200, 100)),
                                                text='Process Data Sets',
                                                manager=manager)
    start_predict = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((240, 295), (160, 40)),
                                                 text='Start Predicting',
                                                 manager=manager)
    wwrite_file = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((240, 350), (130, 40)),
                                               text='Write File(CNN)',
                                               manager=manager)
    NN_comparison = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((500, 200), (200, 100)),
                                                 text='NN Comparison Graph',
                                                 manager=manager)
    CNN_comparison = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((500, 300), (200, 100)),
                                                  text='CNN Comparison Graph',
                                                  manager=manager)
    animated = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((500, 400), (200, 100)),
                                            text='Animated Graph(CNN)',
                                            manager=manager)
    three_D = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((500, 500), (200, 100)),
                                           text='3D Graph(CNN)',
                                           manager=manager)
    trained = None
    Title = 'CSC110 Final Project:'
    Title_2 = 'GMSL Prediction'
    Text0 = 'Period:'
    Text1 = 'Step 1: Process data sets'
    Text2 = 'Step2: Input number of month of prediction.( > 1)'
    Text6 = 'Step4: Only when it says done, click write file.'
    Text7 = 'Step3: Click start prediction.'
    Text3 = 'Step5: Select any graph to see visualizations.'
    Text4 = 'Computing at full speed....'
    Text5 = 'Done!'
    Text8 = 'Welcome!'
    title_font = pygame.font.Font(None, 64)
    base_font = pygame.font.Font(None, 32)
    clock = pygame.time.Clock()
    input_period = ''
    input_rect = pygame.Rect(150, 300, 80, 32)
    rect_color = pygame.Color('lightskyblue3')
    data = None
    period = None
    nn = None
    cnn = None
    background_image = pygame.transform.scale(pygame.image.load("bg.jpg"), (800, 1000))

    while True:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == process_data:
                        print('Data Processed!')
                        data = perform_process_data()
                        print('Data Processed!')
                    if event.ui_element == start_predict:
                        trained = False
                        period = int(input_period)
                        nn = perform_nn(period, data)
                        cnn = perform_cnn(period, data)
                        trained = True
                    if trained is True:
                        if event.ui_element == wwrite_file:
                            write_file_local(period, cnn[4], cnn[3])
                        if event.ui_element == NN_comparison:
                            visualization_2(nn[0])
                        if event.ui_element == CNN_comparison:
                            visualization_1(cnn[0], cnn[2], cnn[1])
                        if event.ui_element == animated:
                            visualization_3(data, period)
                        if event.ui_element == three_D:
                            visualization_4(cnn[4], cnn[3])
            manager.process_events(event)

            if event.type == pygame.KEYDOWN:
                trained = None
                if event.key == pygame.K_BACKSPACE:
                    input_period = input_period[:-1]
                else:
                    input_period += event.unicode

        manager.update(time_delta)
        # set up the background
        surface.blit(background, (0, 0))
        surface.blit(background_image, [0, 0])
        # draw all the elements we need
        ok_surface = base_font.render(Text5, True, (0, 0, 0))
        warn_surface = base_font.render(Text4, True, (0, 0, 0))
        welcome_surface = base_font.render(Text8, True, (0, 0, 0))
        if trained is None:
            surface.blit(welcome_surface, (100, 400))
        elif trained is True:
            surface.blit(ok_surface, (100, 400))
        elif trained is False:
            surface.blit(warn_surface, (100, 400))
        manager.draw_ui(surface)
        title_surface = title_font.render(Title, True, (0, 0, 0))
        surface.blit(title_surface, (50, 50))
        title_2 = title_font.render(Title_2, True, (0, 0, 0))
        surface.blit(title_2, (400, 100))
        pygame.draw.rect(surface, rect_color, input_rect, 2)
        input_surface = base_font.render(input_period, True, (0, 0, 0))
        surface.blit(input_surface, input_rect)
        rule_surface_1 = base_font.render(Text1, True, (0, 0, 0))
        rule_surface_2 = base_font.render(Text2, True, (0, 0, 0))
        rule_surface_3 = base_font.render(Text3, True, (0, 0, 0))
        rule_surface_4 = base_font.render(Text6, True, (0, 0, 0))
        rule_surface_5 = base_font.render(Text7, True, (0, 0, 0))
        surface.blit(rule_surface_1, (100, 600))
        surface.blit(rule_surface_2, (100, 680))
        surface.blit(rule_surface_5, (100, 760))
        surface.blit(rule_surface_4, (100, 840))
        surface.blit(rule_surface_3, (100, 920))
        text_surface_1 = base_font.render(Text0, True, (0, 0, 0))
        surface.blit(text_surface_1, (60, 305))

        pygame.display.update()
