import math


class P1(object):
    def __init__(self):
        self.d0 = 1.0
        self.d1 = 0.0
        self.d2 = 30.0
        self.d3 = 0.0
        self.d4 = 0.0
        self.d5 = 0.0
        self.d7 = 0.0
        self.d8 = 0.0
        self.d9 = 0.0
        self.d11 = 0.0
        self.d12 = 0.0
        self.d14 = 0.0
        self.d15 = 0.0
        self.d16 = 1000.0
        self.d18 = 0.0
        self.d19 = 2.0
        self.d20 = 0.0
        self.d22 = 0.0
        self.d24 = 0.0
        self.d25 = 0.0
        self.d26 = 0.0
        self.d27 = 1.1
        self.d28 = 42164.0
        self.d29 = 0.0
        self.d30 = 0.0
        self.d31 = 1004.0
        self.d32 = 0.0
        self.d33 = 0.0
        self.d35 = 0.0
        self.d36 = 1.5
        self.d37 = 0.0
        self.d38 = 1003.0
        self.d39 = 0.0
        self.d41 = 0.0
        self.d42 = 0.0
        self.d43 = 1002.0
        self.d44 = 0.0
        self.d46 = 0.0
        self.d47 = 1001.0
        self.d48 = 0.0
        self.d50 = 0.0
        self.d52 = 0.0
        self.d53 = 0.0
        self.d54 = 0.0
        self.d55 = 6.457e+06
        self.d57 = 0.0
        self.d59 = 0.0
        self.d60 = -6.35+06
        self.d62 = 0.0
        self.d64 = 0.0
        self.d66 = 0.0
        self.d67 = 0.0
        self.d69 = 0.0
        self.d70 = 0.0
        self.d71 = 0.0
        self.d72 = 0.0
        self.d74 = 0.0
        self.d75 = 0.0
        self.d77 = 0.0
        self.d78 = 8.357e+06
        self.d80 = 0.0
        self.d81 = 6.357e+06
        self.d83 = 0.0
        self.d84 = 6.557e+06
        self.d86 = 0.0
        self.d88 = 0.0
        self.d89 = 0.0
        self.d90 = 0.0
        self.d91 = 0.0
        self.d92 = 0.0
        self.d93 = 0.0
        self.d94 = 0.0
        self.d95 = 0.0
        self.d96 = 6e+24
        self.d98 = 0.0
        self.d99 = 6.67428e-11
        self.d100 = 0.0
        self.d101 = 0.0
        self.d102 = 0.0
        self.d103 = 0.0
        self.d104 = 0.0
        self.d105 = 0.0
        self.d106 = 0.0
        self.d107 = 0.0
        self.d108 = 0.0
        self.d109 = 0.0
        self.d110 = 0.0
        self.d111 = -6922.34
        self.d113 = 0.0
        self.d114 = -4719.32
        self.d116 = 0.0
        self.d117 = -7814.93
        self.d119 = 0.0
        self.d121 = 0.0
        self.d122 = 0.0
        self.d123 = 0.0
        self.d124 = 0.0
        self.d125 = 0.0
        self.d126 = 0.0
        self.d127 = 0.0
        self.d128 = 0.0
        self.d129 = 0.0
        self.d130 = 0.0
        self.d131 = 0.0
        self.d132 = 0.0
        self.d133 = -7875.22
        self.d135 = 0.0
        self.d137 = 0.0
        self.d139 = 0.0
        self.d141 = 0.0
        self.d143 = 0.0
        self.d144 = 0.0
        self.d145 = 0.0
        self.d146 = 0.0
        self.d147 = 0.0
        self.d148 = 0.0
        self.d149 = 0.0
        self.d150 = 0.0
        self.d151 = 0.0
        self.d152 = 0.0
        self.d153 = 0.0
        self.d155 = 0.0
        self.d156 = 0.0
        self.d157 = 0.0
        self.d158 = 0.0
        self.d159 = 0.0
        self.d160 = 0.0
        self.d161 = 0.0
        self.d163 = 0.0
        self.d164 = 0.0
        self.d166 = 0.0
        self.d167 = 0.0
        self.d168 = 0.0
        self.d169 = 0.0
        self.d170 = 0.0
        self.d171 = 0.0
        self.d172 = 0.0
        self.d173 = 0.0
        self.d174 = 0.0
        self.d175 = 0.0
        self.d176 = 0.0
        self.d177 = 0.0
        self.d178 = 0.0
        self.d179 = 0.0
        self.d180 = 0.0
        self.d181 = 0.0
        self.d182 = 0.0
        self.d183 = 0.0
        self.d184 = 0.0
        self.d185 = 0.0
        self.d186 = 0.0
        self.d187 = 0.0
        self.d188 = 0.0
        self.d189 = 1.0
        self.d191 = 0.0
        self.d193 = 0.0
        self.d195 = 0.0
        self.d197 = 0.0
        self.d199 = 0.0
        self.d200 = 0.0
        self.d202 = 0.0
        self.d203 = 0.0
        self.d205 = 0.0
        self.d206 = 0.0
        self.d207 = 0.0
        self.d208 = 0.0
        self.d210 = 0.0
        self.d212 = 0.0
        self.d213 = 0.0
        self.d214 = 10000.0
        self.d216 = 0.0
        self.d217 = 0.0
        self.d218 = 0.0
        self.d219 = 25.0
        self.d220 = 45.0
        self.d221 = 0.0
        self.d222 = 0.0
        self.d223 = 0.0
        self.d224 = 0.0
        self.d225 = 0.0
        self.d226 = 900.0
        self.d227 = 0.0
        self.d228 = 0.0
        self.d230 = 0.0
        self.d231 = 0.0
        self.d232 = 0.0
        self.d234 = 0.0
        self.d235 = 0.0
        self.d237 = 0.0
        self.d238 = 6.357e+06
        self.d239 = 0.0
        self.d241 = 0.0
        self.d242 = 0.0
        self.d248 = 0.0
        self.d249 = 0.0
        self.d250 = 0.0
        self.d251 = 0.0
        self.d252 = 0.0
        self.d253 = 0.0
        self.d254 = 0.0
        self.d255 = 0.0
        self.d256 = 0.0
        self.d257 = 0.0
        self.d258 = 0.0
        self.d259 = 0.0
        self.d260 = 0.0
        self.d261 = 0.0
        self.d262 = 0.0
        self.d263 = 0.0
        self.d264 = 0.0
        self.d265 = 0.0

    def step(self, input, output):
        """
        Make one step of the VM for the problem 1. input and output are the
        lists with port values."""

        self.d1 = self.d265 # copy
        self.d4 = self.d248 # copy
        self.d5 = self.d4 - self.d3 # sub
        self.d7 = (self.d2 if self.d5 == 0 else self.d1) # Phi
        self.d8 = self.d7 - self.d0 # sub
        self.d9 = self.d263 # copy
        self.d11 = (self.d0 if self.d5 == 0 else self.d9) # Phi
        self.d12 = self.d11 - self.d0 # sub
        self.d14 = (self.d8 if self.d12 == 0 else self.d7) # Phi
        self.d15 = self.d264 # copy
        self.d18 = (self.d16 if self.d5 == 0 else self.d15) # Phi
        self.d20 = self.d18 * self.d19 # mult
        self.d22 = (self.d20 if self.d12 == 0 else self.d18) # Phi
        self.d24 = (self.d18 if self.d12 == 0 else self.d12) # Phi
        self.d25 = self.d260 # copy
        self.d26 = self.d262 # copy
        self.d29 = self.d28 * self.d16 # mult
        self.d30 = (self.d29 / self.d27 if self.d27 != 0.0 else 0) # self.div
        self.d32 = input[16000] # input
        self.d33 = self.d32 - self.d31 # sub
        self.d35 = (self.d30 if self.d33 == 0 else self.d3) # Phi
        self.d37 = (self.d29 / self.d36 if self.d36 != 0.0 else 0) # self.div
        self.d39 = self.d32 - self.d38 # sub
        self.d41 = (self.d37 if self.d39 == 0 else self.d35) # Phi
        self.d42 = (self.d29 / self.d19 if self.d19 != 0.0 else 0) # self.div
        self.d44 = self.d32 - self.d43 # sub
        self.d46 = (self.d42 if self.d44 == 0 else self.d41) # Phi
        self.d48 = self.d32 - self.d47 # sub
        self.d50 = (self.d29 if self.d48 == 0 else self.d46) # Phi
        self.d52 = (self.d50 if self.d5 == 0 else self.d26) # Phi
        self.d53 = self.d255 # copy
        self.d57 = (self.d55 if self.d33 == 0 else self.d3) # Phi
        self.d59 = (self.d54 if self.d39 == 0 else self.d57) # Phi
        self.d62 = (self.d60 if self.d44 == 0 else self.d59) # Phi
        self.d64 = (self.d54 if self.d48 == 0 else self.d62) # Phi
        self.d66 = (self.d64 if self.d5 == 0 else self.d53) # Phi
        self.d67 = self.d250 # copy
        self.d69 = (self.d3 if self.d5 == 0 else self.d67) # Phi
        self.d70 = self.d69 - self.d66 # sub
        self.d71 = self.d70 * self.d70 # mult
        self.d72 = self.d249 # copy
        self.d74 = (self.d3 if self.d5 == 0 else self.d72) # Phi
        self.d75 = self.d254 # copy
        self.d77 = (self.d54 if self.d33 == 0 else self.d3) # Phi
        self.d80 = (self.d78 if self.d39 == 0 else self.d77) # Phi
        self.d83 = (self.d81 if self.d44 == 0 else self.d80) # Phi
        self.d86 = (self.d84 if self.d48 == 0 else self.d83) # Phi
        self.d88 = (self.d86 if self.d5 == 0 else self.d75) # Phi
        self.d89 = self.d74 - self.d88 # sub
        self.d90 = self.d89 * self.d89 # mult
        self.d91 = self.d90 + self.d71 # add
        self.d92 = math.sqrt(self.d91) # sqrt
        self.d93 = self.d92 * self.d92 # mult
        self.d94 = self.d93 * self.d92 # mult
        self.d95 = self.d251 # copy
        self.d98 = (self.d96 if self.d5 == 0 else self.d95) # Phi
        self.d100 = self.d99 * self.d98 # mult
        self.d101 = (self.d100 / self.d94 if self.d94 != 0.0 else 0) # self.div
        self.d102 = self.d70 * self.d101 # mult
        self.d103 = input[3] # input
        self.d104 = (self.d0 / self.d0 if self.d0 != 0.0 else 0) # self.div
        self.d105 = (self.d103 / self.d104 if self.d104 != 0.0 else 0) # self.div
        self.d106 = self.d105 + self.d102 # add
        self.d107 = self.d104 * self.d104 # mult
        self.d108 = (self.d107 / self.d19 if self.d19 != 0.0 else 0) # self.div
        self.d109 = self.d106 * self.d108 # mult
        self.d110 = self.d258 # copy
        self.d113 = (self.d111 if self.d39 == 0 else self.d77) # Phi
        self.d116 = (self.d114 if self.d44 == 0 else self.d113) # Phi
        self.d119 = (self.d117 if self.d48 == 0 else self.d116) # Phi
        self.d121 = (self.d119 if self.d5 == 0 else self.d110) # Phi
        self.d122 = self.d121 * self.d104 # mult
        self.d123 = self.d66 + self.d122 # add
        self.d124 = self.d123 + self.d109 # add
        self.d125 = self.d124 - self.d69 # sub
        self.d126 = self.d125 * self.d125 # mult
        self.d127 = self.d89 * self.d101 # mult
        self.d128 = input[2] # input
        self.d129 = (self.d128 / self.d104 if self.d104 != 0.0 else 0) # self.div
        self.d130 = self.d129 + self.d127 # add
        self.d131 = self.d130 * self.d108 # mult
        self.d132 = self.d257 # copy
        self.d135 = (self.d133 if self.d33 == 0 else self.d3) # Phi
        self.d137 = (self.d54 if self.d39 == 0 else self.d135) # Phi
        self.d139 = (self.d114 if self.d44 == 0 else self.d137) # Phi
        self.d141 = (self.d54 if self.d48 == 0 else self.d139) # Phi
        self.d143 = (self.d141 if self.d5 == 0 else self.d132) # Phi
        self.d144 = self.d143 * self.d104 # mult
        self.d145 = self.d88 + self.d144 # add
        self.d146 = self.d145 + self.d131 # add
        self.d147 = self.d146 - self.d74 # sub
        self.d148 = self.d147 * self.d147 # mult
        self.d149 = self.d148 + self.d126 # add
        self.d150 = math.sqrt(self.d149) # sqrt
        self.d151 = self.d150 - self.d52 # sub
        self.d152 = self.d3 - self.d151 # sub
        self.d153 = self.d151 - self.d3 # sub
        self.d155 = (self.d152 if self.d153 < 0 else self.d151) # Phi
        self.d156 = self.d25 + self.d155 # add
        self.d157 = self.d105 * self.d105 # mult
        self.d158 = self.d129 * self.d129 # mult
        self.d159 = self.d158 + self.d157 # add
        self.d160 = math.sqrt(self.d159) # sqrt
        self.d161 = self.d160 - self.d3 # sub
        self.d163 = (self.d156 if self.d161 == 0 else self.d3) # Phi
        self.d164 = self.d155 - self.d16 # sub
        self.d166 = (self.d163 if self.d164 < 0 else self.d3) # Phi
        self.d167 = self.d69 - self.d124 # sub
        self.d168 = self.d167 * self.d167 # mult
        self.d169 = self.d74 - self.d146 # sub
        self.d170 = self.d169 * self.d169 # mult
        self.d171 = self.d170 + self.d168 # add
        self.d172 = math.sqrt(self.d171) # sqrt
        self.d173 = self.d172 * self.d172 # mult
        self.d174 = self.d173 * self.d172 # mult
        self.d175 = (self.d100 / self.d174 if self.d174 != 0.0 else 0) # self.div
        self.d176 = self.d167 * self.d175 # mult
        self.d177 = self.d176 + self.d102 # add
        self.d178 = (self.d177 / self.d19 if self.d19 != 0.0 else 0) # self.div
        self.d179 = self.d105 + self.d178 # add
        self.d180 = self.d179 * self.d104 # mult
        self.d181 = self.d121 + self.d180 # add
        self.d182 = self.d169 * self.d175 # mult
        self.d183 = self.d182 + self.d127 # add
        self.d184 = (self.d183 / self.d19 if self.d19 != 0.0 else 0) # self.div
        self.d185 = self.d129 + self.d184 # add
        self.d186 = self.d185 * self.d104 # mult
        self.d187 = self.d143 + self.d186 # add
        self.d188 = self.d256 # copy
        self.d191 = (self.d189 if self.d33 == 0 else self.d3) # Phi
        self.d193 = (self.d189 if self.d39 == 0 else self.d191) # Phi
        self.d195 = (self.d189 if self.d44 == 0 else self.d193) # Phi
        self.d197 = (self.d189 if self.d48 == 0 else self.d195) # Phi
        self.d199 = (self.d197 if self.d5 == 0 else self.d188) # Phi
        self.d200 = self.d253 # copy
        self.d202 = (self.d3 if self.d5 == 0 else self.d200) # Phi
        self.d203 = self.d252 # copy
        self.d205 = (self.d3 if self.d5 == 0 else self.d203) # Phi
        self.d206 = self.d4 + self.d0 # add
        self.d207 = self.d259 # copy
        self.d208 = self.d207 + self.d0 # add
        self.d210 = (self.d208 if self.d161 == 0 else self.d3) # Phi
        self.d212 = (self.d210 if self.d164 < 0 else self.d3) # Phi
        self.d213 = self.d261 # copy
        self.d216 = (self.d214 if self.d5 == 0 else self.d213) # Phi
        self.d217 = self.d160 * self.d104 # mult
        self.d218 = self.d216 - self.d217 # sub
        self.d221 = self.d214 - self.d218 # sub
        self.d222 = (self.d221 / self.d214 if self.d214 != 0.0 else 0) # self.div
        self.d223 = self.d222 * self.d220 # mult
        self.d224 = self.d7 + self.d223 # add
        self.d225 = self.d224 + self.d219 # add
        self.d227 = (self.d226 / self.d104 if self.d104 != 0.0 else 0) # self.div
        self.d228 = self.d227 - self.d212 # sub
        self.d230 = (self.d225 if self.d228 < 0 else self.d3) # Phi
        self.d231 = self.d218 - self.d3 # sub
        self.d232 = self.d242 - self.d0 # sub
        self.d234 = (self.d232 if self.d231 < 0 else self.d230) # Phi
        self.d235 = self.d214 - self.d217 # sub
        self.d237 = (self.d232 if self.d235 < 0 else self.d234) # Phi
        self.d239 = self.d150 - self.d238 # sub
        self.d241 = (self.d232 if self.d239 < 0 else self.d237) # Phi
        output[0] = self.d241 # out
        output[1] = self.d218 # out
        output[2] = self.d169 # out
        output[3] = self.d167 # out
        output[4] = self.d52 # out
        self.d248 = self.d206 # copy
        self.d249 = self.d74 # copy
        self.d250 = self.d69 # copy
        self.d251 = self.d98 # copy
        self.d252 = self.d205 # copy
        self.d253 = self.d202 # copy
        self.d254 = self.d146 # copy
        self.d255 = self.d124 # copy
        self.d256 = self.d199 # copy
        self.d257 = self.d187 # copy
        self.d258 = self.d181 # copy
        self.d259 = self.d212 # copy
        self.d260 = self.d166 # copy
        self.d261 = self.d218 # copy
        self.d262 = self.d52 # copy
        self.d263 = self.d24 # copy
        self.d264 = self.d22 # copy
        self.d265 = self.d14 # copy

