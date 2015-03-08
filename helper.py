from __future__ import division

from log import Log
from hypothesis import LinearHypothesis, LogisticHypothesis
from cost_function import LinearCostFunction, LogisticCostFunction
from data_reader import LinearRegressionDataReader, LogisticRegressionDataReader
from normaliser import Normaliser
from transformer import Transformer
from gradient_descent import GradientDescent
from denormaliser import Denormaliser

import signal

class Helper:
    def __init__(self, file_name):
        self.file_name = file_name
        self.err_check = False
        self.learning_rate = 1
        self.regularisation_coefficient = 0
        self.max_iterations = 1000
        self.with_linear = False
        self.other_terms = {}
        self.hypothesis_class = None
        self.cost_fn_class = None
        self.data_reader_class = None
        self.output_value_checker = None
        self.normalise_data = True
        self.test_on_completion = False

    def with_error_checking(self, val):
        self.err_check = val
        return self

    def with_learning_rate(self, val):
        self.learning_rate = val
        return self

    def with_regularisation_coefficient(self, val):
        self.regularisation_coefficient = val
        return self

    def with_iterations(self, val):
        self.max_iterations = val
        return self

    def with_linear_terms(self):
        self.with_linear = True
        return self

    def with_term(self, name, fn):
        self.other_terms[name] = fn
        return self

    def with_linear_regression(self):
        self.hypothesis_class = LinearHypothesis
        self.cost_fn_class = LinearCostFunction
        self.data_reader_class = LinearRegressionDataReader
        return self

    def with_logistic_regression(self):
        self.hypothesis_class = LogisticHypothesis
        self.cost_fn_class = LogisticCostFunction
        self.data_reader_class = LogisticRegressionDataReader
        return self

    def with_normalisation(self, val):
        self.normalise_data = val
        return self

    def with_test_on_completion(self):
        self.test_on_completion = True
        return self

    def go(self):
        log = Log()

        raw_data_lines = open(self.file_name).readlines()

        reader = self.data_reader_class(raw_data_lines)

        raw_data_inputs     = reader.input_values
        raw_data_outputs    = reader.output_values
        accepted_line_count = reader.accepted_count
        rejected_lines      = reader.rejected_lines
        raw_input_var_count = reader.input_var_count

        # Display results of reading data file
        for rejected_line in rejected_lines:
            log.error('Bad input line: ' + rejected_line)

        log.bar()
        log.info('Read %s, loaded %s lines, rejected %s, %s input values per line' % (self.file_name, accepted_line_count, len(rejected_lines), raw_input_var_count))
        log.info('Press Ctrl-C at any time to stop working and show results')
        log.bar()

        # Convert the raw inputs using the transformer functions
        transformer = Transformer(raw_data_inputs)

        if (self.with_linear):
            transformer.add_linear_terms()

        for name, fn in self.other_terms.iteritems():
            transformer.add_new_term(name, fn)

        raw_variables = transformer.variables

        # Apply Feature Scaling and Mean Normalisation
        normaliser = Normaliser()
        normalised_variables = map(normaliser.normalise, raw_variables)

        hypothesis    = self.hypothesis_class(len(normalised_variables))
        cost_function = self.cost_fn_class(raw_data_outputs, self.learning_rate, self.regularisation_coefficient)

        gradient_descent = GradientDescent(hypothesis, cost_function, normalised_variables, raw_data_outputs)
        signal.signal(signal.SIGINT, gradient_descent.interrupt)

        gradient_descent.set_iterations(self.max_iterations)
        if self.err_check:
            gradient_descent.set_error_checking()

        gradient_descent.calculate()

        # Denormalise the calculated theta values
        normalised_thetas = gradient_descent.hypothesis.theta_values
        denormaliser = Denormaliser()
        final_thetas = denormaliser.denormalise(normalised_thetas, normalised_variables)

        # Run hypothesis against original values
        if self.test_on_completion:
            log.bar()
            hypothesis.theta_values = final_thetas

            raw_variable_data = zip(*map(lambda v : v.data, raw_variables))
            for i, o in zip(raw_variable_data, raw_data_outputs):
                log.info('{0:>8} .... {1: .8f}'.format(o, hypothesis.calculate(i)))

        # Display results
        log.bar()
        log.info('Theta values:')
        log.underline()
        for nv, ht in zip(normalised_variables, final_thetas):
            log.info("{:>8} = {:>16.8f}".format(nv.variable.name, ht))

        log.info('')
        log.info('Completed %s iterations' % (gradient_descent.iterations,))
        log.bar()