# Load a binary file and output it as a C array in a header file
# Set the variables:
# BIN_TO_H_OUTPUT: output header file name
# BIN_TO_H_BINARY_FILE: input binary file
# BIN_TO_H_VAR_NAME: variable name for the array
function(bin_to_h)
    file(READ ${BIN_TO_H_BINARY_FILE} BINARY_CONTENT HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ARRAY_CONTENT ${BINARY_CONTENT})
    # Remove the extra comma at the end
    string(REGEX REPLACE ",$" "" ARRAY_CONTENT ${ARRAY_CONTENT})
    file(WRITE ${BIN_TO_H_OUTPUT} "#pragma once\n")
    file(APPEND ${BIN_TO_H_OUTPUT} "const uint8_t ${BIN_TO_H_VAR_NAME}[] = {\n")
    file(APPEND ${BIN_TO_H_OUTPUT} ${ARRAY_CONTENT})
    file(APPEND ${BIN_TO_H_OUTPUT} "\n};")
endfunction()

bin_to_h()

