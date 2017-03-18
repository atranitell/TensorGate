
checkpoint_exclude_scopes = 'inceptionv1,inceptionv2'
print([scope.strip() for scope in checkpoint_exclude_scopes.split(',')])