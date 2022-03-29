import React from 'react';
import { Link } from 'react-router-dom';
import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import Divider from '@material-ui/core/Divider';
import { withStyles } from '@material-ui/core/styles';
import ListItemText from '@material-ui/core/ListItemText';
import { PROJECT_TYPES } from './constants';

const drawerWidth = 150;

const styles = theme => ({
    drawer: {
        width: drawerWidth,
        flexShrink: 0
    },
    listItemRoot1: {
        "&.MuiListItem-root": {
            backgroundColor: 'red',
            color: 'white'
        }
    },
    listItemRoot2: {
        "&.MuiListItem-root": {
            backgroundColor: 'green',
            color: 'white'
        }
    },
});


class SideBar extends React.Component {

    render() {
        const { classes, ...rest } = this.props;

        return (
            <nav className={classes.drawer}>
                <Drawer
                    variant="permanent"
                >
                    <List>
                        {this.props.projectName &&
                            <React.Fragment>
                                <ListItem
                                    button
                                    classes={{ root: classes.listItemRoot1 }}
                                    component={Link}
                                    to={`/projects/${this.props.projectNameSlug}`}
                                >
                                    <ListItemText>
                                        FINISH ITER
                                    </ListItemText>
                                </ListItem>
                                <Divider />
                            </React.Fragment>
                        }
                        <ListItem
                            button
                            classes={{ root: classes.listItemRoot2 }}
                            component={Link}
                            to="/projects">
                            <ListItemText>
                                Projects
                            </ListItemText>
                        </ListItem>
                    </List>
                </Drawer>
            </nav>
        )
    }
}

export default withStyles(styles)(SideBar);
export { drawerWidth };