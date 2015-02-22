class DataReader:
    def __init__(self, lines):
        inputs   = []
        outputs  = []
        rejected = []
        ok_count = 0
        n = None
        
        def is_comment(line):
            return line[0] == '#'

        for line in map(str.strip, lines):
            if is_comment(line):
                continue

            line_err = False
            parts = line.split(':')

            if len(parts) != 2:
                line_err = True
            else:
                try:
                    result_part, data_part = parts
                    data_parts = data_part.split(',')

                    if n is None:
                        n = len(data_parts)
                    
                    if n == len(data_parts):
                        inputs.append(map(float, data_parts))
                        outputs.append(float(result_part))

                    else:
                        line_err = True    

                except ValueError:
                    line_err = True

            if line_err:
                rejected.append(line)
            else:
                ok_count += 1

        self.input_values    = inputs
        self.output_values   = outputs
        self.accepted_count  = ok_count
        self.rejected_lines  = rejected
        self.input_var_count = n
